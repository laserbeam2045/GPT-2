#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
from flask import Flask, request, Response
from flask_sslify import SSLify
import json
import os
import time
import copy

app = Flask(__name__)
#sslify = SSLify(app)

seed = None
length = 2
width = 1
top_k_num = 20
batch_size = width * top_k_num
temperature = 1
top_k = 0
top_p = 1
threshold1 = 0.00001
threshold2 = 0.4
maxlen = 100
model_name = '124M'
#model_name = '355M'
#model_name = '774M'
#model_name = '1558M'
models_dir = 'models'
ckpts_dir = 'tf_ckpts'
models_dir = os.path.expanduser(os.path.expandvars(models_dir))
ckpts_dir = os.path.expanduser(os.path.expandvars(ckpts_dir))

#var = tf.train.list_variables(tf.train.latest_checkpoint(os.path.join(models_dir, model_name)))
#print(var)


@app.before_first_request
def load_model():
    import numpy as np
    import tensorflow as tf
    import model, sample, encoder
    #from src import model, sample, encoder

    print("\n================================")
    print("Loading Model...")
    print("================================")

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")

    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    app.encoder = encoder.get_encoder(model_name, models_dir)
    app.pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

    app.gpt2 = {}
    app.gpt2['short'] = sample.GPT_2(
        hparams=hparams,
        length=1,
        batch_size=1,
        temperature=temperature,
        top_k=top_k, top_p=top_p,
        threshold=threshold2,
    )
    app.gpt2['multi'] = sample.GPT_2(
        hparams=hparams,
        length=length,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k, top_p=top_p,
        threshold = threshold2,
    )
    app.gpt2['long'] = sample.GPT_2(
        hparams=hparams,
        length=5,
        batch_size=1,
        temperature=temperature,
        top_k=top_k, top_p=top_p,
        threshold = threshold2
    )
    #tf.executing_eagerly()

    # 内部変数を作成させるために空打ちする
    predict('initialize', 'short')
    predict(['initialize' for _ in range(batch_size)], 'multi')
    predict('initialize', 'long')

    # load model as keras
    app.gpt2['short'].m.load_weights(os.path.join(ckpts_dir, model_name, 'model'))
    app.gpt2['multi'].m.load_weights(os.path.join(ckpts_dir, model_name, 'model'))
    app.gpt2['long'].m.load_weights(os.path.join(ckpts_dir, model_name, 'model'))
    #app.gpt2_multi.m.save_weights(os.path.join(ckpts_dir, model_name, 'model'))

    predict('initialize', 'short')
    predict(['initialize' for _ in range(batch_size)], 'multi')
    predict('initialize', 'long')
    
    # load model as tensorflow
    #restore = tf.train.Checkpoint(model=app.gpt2_single.m)
    #status = restore.restore(tf.train.latest_checkpoint(os.path.join(models_dir, model_name)))
    #app.gpt2_single.m.save_weights(os.path.join(ckpts_dir, model_name, 'model'))

    #restore.listed = []
    #for var in app.gpt_2.m.variables:
    #    restore.listed.append(var)
    #manager = tf.train.CheckpointManager(restore,  './tf_ckpts/'+model_name, max_to_keep=1)
    #save_path = manager.save()
    print("\n================================")
    print("Model Loaded.")
    print("================================")
    #print(tf.autograph.to_code(app.gpt_2.m.__call__.python_function))


def predict(input, model_name):

    if type(input) == str:
        context = [app.encoder.encode(input)]
        context = app.pad_sequences(context, maxlen=maxlen)
        tokens, probs = app.gpt2[model_name].predict(context)
    elif type(input) == list:
        context = [app.encoder.encode(text) for text in input]
        context = app.pad_sequences(context, maxlen=maxlen)
        tokens, probs = app.gpt2[model_name].predict(context)

    return tokens.numpy(), probs.numpy()


# 入力途中の可能性のある単語を絞り込む関数
# 戻り値：3次元のリスト
def get_words_list(tokens, probs, partial_match, parfect_match):

    # 予測された単語が、末尾の文字列と完全一致した場合、入力途中ではない可能性が高いと考える
    if len(parfect_match) and tokens[0][0] == parfect_match[0]:
        print("It will not be halfway word: {}".format(app.encoder.decode([tokens[0][0]])))
        return []

    words_list = []
    for token in partial_match:
        prob = probs[0][0][token]
        if prob < threshold1: continue
        word = app.encoder.decode([token]).strip()
        words_list.append([prob, word, True])    # 確率、単語、includeFlag

    words_list.sort(reverse=True)
    words_list = list(map(lambda x:[[str(x[0]), x[1], x[2]]], words_list))

    return words_list


# バッチ処理用の入力データを作成する関数
def create_batch(predicted_words, prev_input_text):
    # 確率上位のtop_k_num個の単語をwidth個ずつ先頭に増やす
    stack = []
    for word in predicted_words[:top_k_num]:
        for _ in range(width):
            word = copy.deepcopy(word)
            stack.append(word)
    predicted_words = stack + predicted_words[top_k_num:]
    
    # 数がbatch_sizeに満たない場合はその分を先頭の単語だけで埋める
    lack = batch_size - len(predicted_words)
    for _ in range(lack):
        predicted_words.insert(0, copy.deepcopy(predicted_words[0]))

    # 末尾の文字列を除いた文字列と、確率上位の単語を結合する
    input = []
    for i in range(batch_size):
        next_word = predicted_words[i][0][1]
        input.append(prev_input_text + ' ' + next_word)

    return input, predicted_words


def get_texts(tokens, probs):
    texts = []
    for i, token in enumerate(tokens):
        prob = probs[i][token]
        text = app.encoder.decode([token])
        if prob > threshold2:
            if len(texts) and joinable(text):
                texts[-1][1] += text
            else:
                texts.append([str(prob), text, False])
        else:
            break

    return texts


def joinable(text):
    return (
        text[0] != ' ' and text[0] != '\'' and
        text[0] != ',' and text[0] != '.'
    )


def print_elapsed_time(start_time, predicted_words, last_word):
    print("\n================================")
    print("last_word: " + last_word)
    for i, pred in enumerate(predicted_words):
        print(pred)
        if i >= batch_size: break
    elapsed_time = int((time.time() - start_time) * 1000) / 1000
    print("elapsed_time: {0}[sec]".format(elapsed_time))
    print("================================")


def make_response(params):
    headers = {
        'Content-type': 'application/json; charset=utf-8',
        'Access-Control-Allow-Origin': '*',
    }
    response = json.dumps(params).encode("utf-8")
    return Response(response=response, status=200, headers=headers)


def parse(request):
    charset = request.mimetype_params.get('charset') or 'UTF-8'
    data = json.loads(request.get_data().decode(charset, 'replace'))
    
    return data


def missmatch(word):
    return (
        word[0] == ',' or
        word[0] == '.' or
        word[0] == '\'' or
        word[0] == '"'
    )


@app.route("/", methods=["POST"])
def api_model():

    start_time = time.time()
    data = parse(request)
    prev_input_text = data['text'][:-(len(data['lastWord']) + 1)]
    predicted_words = []

    # テキストの末尾の文字列を先頭に含む単語が、辞書中に存在する場合、
    # その単語を入力途中であると仮定して、末尾の文字列を除いたテキストを入力とし推論する。
    # その結果得られた確率を基に、確率が閾値より高い単語だけに絞り込み、次の推論に繋げる。
    if not data['tabFlag'] and not data['blankFlag'] and prev_input_text:
        partial_match, parfect_match = app.encoder.get_match_code(data['lastWord'])
        if len(partial_match):
            tokens, probs = predict(prev_input_text, 'short')
            predicted_words = get_words_list(tokens, probs, partial_match, parfect_match)

    if len(predicted_words):
        input, predicted_words = create_batch(predicted_words, prev_input_text)
        tokens, probs = predict(input, 'multi')
        for i in range(batch_size):
            texts = get_texts(tokens[i], probs[i])
            if len(texts):
                if joinable(texts[0][1]):
                    predicted_words[i][0][1] += texts[0][1]
                    if len(texts) <= 1: continue
                    predicted_words[i] = ([predicted_words[i][0]] + texts[1:])
                else:
                    predicted_words[i] = ([predicted_words[i][0]] + texts)
    else:
        tokens, probs = predict(data['text'], 'long')
        texts = get_texts(tokens[0], probs[0])
        if len(texts) and not (data['blankFlag'] and missmatch(texts[0][0][1])) :
            predicted_words.append(texts)

    print_elapsed_time(start_time, predicted_words, data['lastWord'])
    return make_response({'predictedWords': predicted_words})


if __name__ == '__main__':
    print("Serving...")
    #app.run(host="0.0.0.0", port=9999)
    app.run()