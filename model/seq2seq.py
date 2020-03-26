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
from util import str2id, id2str


class ScaleShift(Layer):
    # 缩放平移变换层
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs, **kwargs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


class OurLayer(Layer):
    # 定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                layer.build(K.int_shape(args[0]))
            else:
                layer.build(K.int_shape(kwargs['inputs']))

            self._trainable_weights.extend(layer._trainable_weights)
            self._non_trainable_weights.extend(layer._non_trainable_weights)
        return layer.call(*args, **kwargs)


class OurBidirectional(OurLayer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """

    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = copy.deepcopy(layer)
        self.backward_layer = copy.deepcopy(layer)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)

    def call(self, inputs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], 2)
        return x * mask

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.forward_layer.units * 2)


class SelfModulatedLayerNormalization(OurLayer):
    # 模仿Self-Modulated Batch Normalization，只不过降Batch Normalization改为Layer Normalization

    def __init__(self, num_hidden, **kwargs):
        super(SelfModulatedLayerNormalization, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        super(SelfModulatedLayerNormalization, self).build(input_shape)
        output_dim = input_shape[0][-1]
        self.layernorm = LayerNormalization(center=False, scale=False)
        self.beta_dense_1 = Dense(self.num_hidden, activation='relu')
        self.beta_dense_2 = Dense(output_dim)
        self.gamma_dense_1 = Dense(self.num_hidden, activation='relu')
        self.gamma_dense_2 = Dense(output_dim)

    def call(self, inputs):
        inputs, cond = inputs
        inputs = self.reuse(self.layernorm, inputs)
        beta = self.reuse(self.beta_dense_1, cond)
        beta = self.reuse(self.beta_dense_2, beta)
        gamma = self.reuse(self.gamma_dense_1, cond)
        gamma = self.reuse(self.gamma_dense_2, gamma)
        for _ in range(K.ndim(inputs) - K.ndim(cond)):
            beta = K.expand_dims(beta, 1)
            gamma = K.expand_dims(gamma, 1)
        return inputs * (gamma + 1) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Attention(OurLayer):
    # 多头注意力机制
    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs, **kwargs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(a[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class Evaluate(Callback):
    def __init__(self, model):
        super(Evaluate, self).__init__()
        self.lowest = 1e10
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            self.model.save_weights('./best_model_epoch={}_loss={}.weights'.format(epoch, self.lowest))


class Seq2seq():
    def __init__(self, chars, char2id, id2char, char_size, z_dim):
        self.chars = chars
        self.char2id = char2id
        self.id2char = id2char
        self.char_size = char_size
        self.z_dim = z_dim
        # self.epochs = epochs
        # self.batch_size = batch_size
        # self.lr = lr
        # self.steps_per_epoch = steps_per_epoch

    def train(self, X_train, Y_train, epochs, batch_size, lr=1e-3, steps_per_epoch=1000):
        model = self.network()
        # self.lr = LearningRateScheduler(self.scheduler(epochs, model))
        model.compile(optimizer=Adam(lr))
        evaluator = Evaluate(model=model)

        model.fit_generator(self.data_generator(X_train, Y_train, batch_size),
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            callbacks=[evaluator])

    def predict(self, s, model_weights, topk=3, maxlen=64):
        model = self.network()
        model.load_weights(model_weights)
        # beam search解码 :每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索

        xid = np.array([str2id(s, self.char2id)] * topk)  # 输入转id
        yid = np.array([[2]] * topk)  # 解码均以<start>开头，这里<start>的id为2
        scores = [0] * topk  # 候选答案分数
        for i in range(maxlen):  # 强制要求输出不超过maxlen字
            proba = model.predict([xid, yid])[:, i, 3:]  # 直接忽略<padding>、<unk>、<start>
            log_proba = np.log(proba + 1e-6)  # 取对数，方便计算
            arg_topk = log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
            _yid = []  # 暂存的候选目标序列
            _scores = []  # 暂存的候选目标序列得分
            if i == 0:
                for j in range(topk):
                    _yid.append(list(yid[j]) + [arg_topk[0][j] + 3])
                    _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
            else:
                for j in range(topk):
                    for k in range(topk):  # 遍历topk*topk的组合
                        _yid.append(list(yid[j]) + [arg_topk[j][k] + 3])
                        _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
                _arg_topk = np.argsort(_scores)[-topk:]  # 从中选出新的topk
                _yid = [_yid[k] for k in _arg_topk]
                _scores = [_scores[k] for k in _arg_topk]
            yid = np.array(_yid)
            scores = np.array(_scores)
            ends = np.where(yid[:, -1] == 3)[0]
            if len(ends) > 0:
                k = ends[scores[ends].argmax()]
                return id2str(yid[k], self.id2char)
        # 如果maxlen字都找不到<end>，直接返回
        return id2str(yid[np.argmax(scores)], self.id2char)

    def network(self, x_in=Input(shape=(None,)), y_in=Input(shape=(None,))):
        # 搭建seq2seq模型
        x, y = x_in, y_in
        x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
        y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)

        x_one_hot = Lambda(self.to_one_hot)([x, x_mask])
        x_prior = ScaleShift()(x_one_hot)  # 学习输出的先验分布（标题的字词很可能在文章出现过）

        embedding = Embedding(len(self.chars) + 4, self.char_size)
        x = embedding(x)
        y = embedding(y)

        # encoder，双层双向LSTM
        x = LayerNormalization()(x)
        x = OurBidirectional(LSTM(self.z_dim // 2, return_sequences=True))([x, x_mask])
        x = LayerNormalization()(x)
        x = OurBidirectional(LSTM(self.z_dim // 2, return_sequences=True))([x, x_mask])
        x_max = Lambda(self.seq_maxpool)([x, x_mask])

        # decoder，双层单向LSTM
        y = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = LSTM(self.z_dim, return_sequences=True)(y)
        y = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = LSTM(self.z_dim, return_sequences=True)(y)
        y = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])

        # attention交互
        xy = Attention(8, 16)([y, x, x, x_mask])
        xy = Concatenate()([y, xy])

        # 输出分类
        xy = Dense(self.char_size)(xy)
        xy = LeakyReLU(0.2)(xy)
        xy = Dense(len(self.chars) + 4)(xy)
        xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])  # 与先验结果平均
        xy = Activation('softmax')(xy)

        # 交叉熵作为loss，但mask掉padding部分
        cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
        cross_entropy = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

        model = Model([x_in, y_in], xy)
        model.add_loss(cross_entropy)
        return model

    # 学习率控制
    def scheduler(self, epochs, model):
        # 每隔15个epoch，学习率减小为原来的1/2
        if epochs % 15 == 0 and epochs != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(model.optimizer.lr)

    def to_one_hot(self, x):
        # 转one_hot  输出一个词表大小的向量，来标记该词是否在文章出现过
        x, x_mask = x
        x = K.cast(x, 'int32')  # 相当于转换类型
        x = K.one_hot(x, len(self.chars) + 4)  # 转one_hot
        x = K.sum(x_mask * x, 1, keepdims=True)
        x = K.cast(K.greater(x, 0.5), 'float32')  # 相当于那个先验知识
        return x

    def seq_avgpool(self, x):
        # seq是[None, seq_len, s_size]的格式，mask是[None, seq_len, 1]的格式，先除去mask部分， 然后再做avgpooling。
        seq, mask = x
        return K.sum(seq * mask, 1) / (K.sum(mask, 1) + 1e-6)

    def seq_maxpool(self, x):
        # seq是[None, seq_len, s_size]的格式，mask是[None, seq_len, 1]的格式，先除去mask部分，然后再做maxpooling。
        seq, mask = x
        seq -= (1 - mask) * 1e10
        return K.max(seq, 1)

    def padding(self, x):
        # padding到batch内最大长度
        ml = max([len(i) for i in x])
        return [i + [0] * (ml - len(i)) for i in x]  # 长度不够这填充0

    def data_generator(self, X_train, Y_train, batch_size):
        # 数据生成器
        X, Y = [], []
        while True:
            for c, t in zip(X_train, Y_train):
                X.append(str2id(c, self.char2id))
                Y.append(str2id(t, self.char2id, start_end=True))  # 只需给标题加开始和结尾
                if len(X) == batch_size:
                    X = np.array(self.padding(X))
                    Y = np.array(self.padding(Y))
                    yield [X, Y], None
                    X, Y = [], []
