# coding=utf-8
import os
import json
import numpy as np
import keras.backend as K
from keras.layers import Layer
import copy
from keras_layer_normalization import LayerNormalization
from keras.layers import Input, Lambda, Embedding, LSTM, LeakyReLU, Concatenate, Activation
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback


def load_data(path):
    titles = []
    texts = []
    # 加载数据 并将数据标题和新闻分别存到titles和texts中
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.replace('\n', '').split(',')
            titles.append(line[0])
            texts.append(line[-1])
    return titles, texts





def load_all_data(base_path):
    path_list = os.listdir(base_path)
    # print(path_list)
    titles = []
    texts = []
    for path in path_list:
        print(base_path + path)
        titles_temp, texts_temp = load_data(base_path + path)
        titles += titles_temp
        texts += texts_temp
    return titles, texts




def word2vec(titles, texts,min_count=32):
    # 加载字到id的映射， 或者整理字到id的映射
    if os.path.exists('seq2seq_config.json'):
        chars, id2char, char2id = json.load(open('seq2seq_config.json'))
        id2char = {int(i): j for i, j in id2char.items()}

    else:
        chars = {}
        for t in titles:
            for w in t:
                chars[w] = chars.get(w, 0) + 1
        for c in texts:
            for w in c:
                chars[w] = chars.get(w, 0) + 1

        chars = {i: j for i, j in chars.items() if j >= min_count}  # 过滤低频次

        # 0:mask, 1:unk, 2:start, 3:end

        id2char = {i + 4: j for i, j in enumerate(chars)}
        char2id = {j: i for i, j in id2char.items()}
        json.dump([chars, id2char, char2id], open('seq2seq_config.json', 'w'))
    return chars, id2char, char2id


def str2id(s, char2id, maxlen=400,start_end=False):
    if start_end:  # 若是开始或结束补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen - 2]]  # 转语料为id
        ids = [2] + ids + [3]  # 加头尾
    else:
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids, id2char):
    # 数字转汉字，没有填充''
    return ''.join([id2char.get(i, '') for i in ids])
