from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from training.hparam import HParams
#from src.training.hparam import HParams

def default_hparams():
    #return tf.contrib.training.HParams(
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


"""Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
class Norm(layers.Layer):
    def __init__(self, x, **kwargs):
        super(Norm, self).__init__(**kwargs)
        n_state = x.shape[-1]
        g_init = tf.constant_initializer(1)
        b_init = tf.constant_initializer(0)
        self.g = tf.Variable(g_init(dtype=x.dtype, shape=[n_state]), name='g')
        self.b = tf.Variable(b_init(dtype=x.dtype, shape=[n_state]), name='b')

    def __call__(self, x, *, axis=-1, epsilon=1e-5):
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + epsilon)
        x = x*self.g + self.b
        return x


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])


class Conv1d(layers.Layer):
    def __init__(self, x, nf, *, w_init_stdev=0.02, **kwargs):
        super(Conv1d, self).__init__(**kwargs)
        *start, nx = shape_list(x)
        w_init = tf.random_normal_initializer(stddev=w_init_stdev)
        b_init = tf.constant_initializer(0)
        self.w = tf.Variable(w_init(dtype=x.dtype, shape=[1, nx, nf]), name='w')
        self.b = tf.Variable(b_init(dtype=x.dtype, shape=[nf]), name='b')

    def __call__(self, x, nf):
        *start, nx = shape_list(x)
        a = tf.reshape(x, [-1, nx])
        b = tf.reshape(self.w, [-1, nf])
        c = tf.reshape(tf.matmul(a, b)+self.b, start+[nf])
        return c


def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    #i = tf.reshape(tf.range(nd), [nd, 1])
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


class Attn(layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Attn, self).__init__(**kwargs)
        self.hparams = hparams
        self.c_attn = None
        self.c_proj = None

    def split_heads(self, x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, self.hparams.n_head), [0, 2, 1, 3])

    def merge_heads(self, x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(self, w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(self, q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.math.rsqrt(tf.cast(v.shape[-1], w.dtype))

        w = self.mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    def __call__(self, x, n_state, *, past, hparams):
        assert x.shape.ndims == 3  # Should be [batch, sequence, features]
        assert n_state % hparams.n_head == 0
        if past is not None:
            assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

        if self.c_attn is None:
            self.c_attn = Conv1d(x, n_state*3, name='c_attn')
        c = self.c_attn(x, n_state*3)
        q, k, v = map(self.split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = self.multihead_attn(q, k, v)
        a = self.merge_heads(a)
        if self.c_proj is None:
            self.c_proj = Conv1d(a, n_state, name='c_proj')
        a = self.c_proj(a, n_state)
        return a, present


class Mlp(layers.Layer):
    def __init__(self, **kwargs):
        super(Mlp, self).__init__(**kwargs)
        self.c_fc = None
        self.c_proj = None

    def __call__(self, x, n_state):
        nx = x.shape[-1]
        if self.c_fc is None:
            self.c_fc = Conv1d(x, n_state, name='c_fc')
        h = gelu(self.c_fc(x, n_state))
        if self.c_proj is None:
            self.c_proj = Conv1d(h, nx, name='c_proj')
        h2 = self.c_proj(h, nx)
        return h2


class Block(layers.Layer):
    def __init__(self, x, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.attn = None
        self.mlp = None
        self.norm_ln_1 = Norm(x, name='ln_1')
        self.norm_ln_2 = Norm(x, name='ln_2')

    def __call__(self, x, *, past, hparams):
        nx = x.shape[-1]
        if self.attn is None:
            self.attn = Attn(hparams=hparams, name='attn')
        a, present = self.attn(self.norm_ln_1(x), nx, past=past, hparams=hparams)
        x = x + a
        if self.mlp is None:
            self.mlp = Mlp(name='mlp')
        m = self.mlp(self.norm_ln_2(x), nx*4)
        x = x + m
        return x, present


def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


class Model(tf.keras.Model):
    def __init__(self, hparams, name='model', **kwargs):
        super(Model, self).__init__(name=name, **kwargs)
        wpe_shape = [hparams.n_ctx,   hparams.n_embd]
        wte_shape = [hparams.n_vocab, hparams.n_embd]
        wpe_init = tf.random_normal_initializer(stddev=0.01)
        wte_init = tf.random_normal_initializer(stddev=0.02)
        self.layers_list = []
        self.norm = None
        self.wpe = tf.Variable(wpe_init(shape=wpe_shape), name="wpe")
        self.wte = tf.Variable(wte_init(shape=wte_shape), name="wte")

    def __call__(self, X, hparams, past=None):
        results = {}
        batch, sequence = shape_list(X)

        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(self.wte, X) + tf.gather(self.wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer

        for layer, past in enumerate(pasts):

            if layer + 1 > len(self.layers_list):
                self.layers_list.append(Block(h, name='h%d' % layer))
            h, present = self.layers_list[layer](h, past=past, hparams=hparams)
            presents.append(present)

        results['present'] = tf.stack(presents, axis=1)

        if self.norm is None:
            self.norm = Norm(h, name='ln_f')
        h = self.norm(h)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, self.wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results