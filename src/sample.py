from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import model
#from src import model

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.reshape(tf.gather_nd(sorted_logits, indices), [batch, 1])
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


class GPT_2(tf.Module):
    def __init__(self, *, hparams, length, batch_size=None, temperature=1, top_k=0, top_p=1, threshold=0.5):
        super(GPT_2, self).__init__()

        #if start_token is None:
        #    assert context is not None, 'Specify exactly one of start_token and context!'
        #else:
        #    assert context is None, 'Specify exactly one of start_token and context!'
        #    context = tf.fill([batch_size, 1], start_token)
        self.m = model.Model(hparams)
        self.input = None

        self.hparams = hparams
        self.length = length
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.threshold = threshold

    def step(self, tokens, past=None):
        lm_output = self.m(X=tokens, hparams=self.hparams, past=past)

        logits = lm_output['logits'][:, :, :self.hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=self.hparams, batch_size=self.batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    def body(self, past, prev, output, prev_probs):
        next_outputs = self.step(prev, past=past)
        logits = next_outputs['logits'][:, -1, :] / tf.cast(self.temperature, tf.float32)
        logits = tf.reshape(logits, [self.batch_size, 50257]) / tf.cast(self.temperature, tf.float32)
        logits = top_k_logits(logits, k=self.top_k)
        logits = top_p_logits(logits, p=self.top_p)
        #if self.batch_size == 1:
        samples = tf.argmax(input=logits, axis=1, output_type=tf.int32)
        samples = tf.reshape(samples, [self.batch_size, 1])
        #else:
            #samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            #samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
        probs = model.softmax(logits)
        probs = tf.reshape(probs, [self.batch_size, 1, self.hparams.n_vocab])
        return [
            next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
            samples,
            tf.concat([output, samples], axis=1),
            probs if prev_probs is None else tf.concat([prev_probs, probs], axis=1)
        ]

    @tf.function
    def predict(self, context):
        if self.input is None:
            self.input = tf.Variable(context)
        else:
            self.input.assign(context)

        past, prev, output, probs = self.body(None, self.input, self.input, None)

        def cond(*args):
            if self.length == 1:
                return False
            elif self.batch_size != 1:
                return True
            else:
                x = tf.cast(self.threshold, tf.float32)
                token = output[0][-1]
                y = probs[0][-1][token]
                return tf.math.less(x, y)

        _, _, output, probs = tf.while_loop(
            cond=cond, body=self.body,
            maximum_iterations=self.length - 1,
            loop_vars=[
                past,
                prev,
                output,
                probs,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=self.hparams, batch_size=self.batch_size)),
                tf.TensorShape([self.batch_size, None]),
                tf.TensorShape([self.batch_size, None]),
                tf.TensorShape([self.batch_size, None, self.hparams.n_vocab]),
            ],
            back_prop=False,
        )
        #print(self.m.variables)

        return output[:, self.input.shape[-1]:], probs