import tensorflow as tf
import collections
import math
import os
import random
import zipfile
import pickle
import numpy as np
from six.moves import urllib
from collections import defaultdict
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import gensim as gensim
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense,GRU
from keras.optimizers import Adam


# 1. 下载数据集
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    # 获取文件相关属性
    statinfo = os.stat(filename)
    # 比对文件的大小是否正确
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename
filename = maybe_download('text8.zip', 31344016)

# 2.训练word2vec模型
# sentences = gensim.models.word2vec.LineSentence('./text8')
#
# wv_model =gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
#
# #wv_model.save('./word2vecModel')
#加载模型
wv_model = gensim.models.Word2Vec.load('./word2vecModel')

#comouter的vec
print(wv_model.wv['computer'] )
#
# #计算词的关系
# print(wv_model.wv.most_similar(positive=['woman', 'king'], negative=['man']))
#
# print(wv_model.wv.most_similar(positive=['boy', 'girl'], negative=['man']))
#
# print(wv_model.wv.similarity('woman', 'man'))
# print(wv_model.wv.similarity('boy', 'girl'))



#filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    print(len(data))
    return data


# 单词表
words = read_data(filename)

print(words[1])



#3.进行rgu训练
#3.1定义训练的参数
TIME_STEPS = 50     # same as the height of the image
INPUT_SIZE = 100     # same as the width of the image
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 100
CELL_SIZE = 50
LR = 0.001

#3.2定义获取数据集的方法
def get_train_dataset(words):
    # 去掉只出现一次的单词
    frequency = defaultdict(int)
    for token in words:
        frequency[token] += 1
    words = [token for token in words if frequency[token] > 1]

    size = len(words)
    print(size)
    x_train = []
    y_train = []
    X_train =[]
    Y_train = []

    size = int(size/51)-1
    j=0
    for i in range(0,size,1):
        x_train.append(words[i*51:i*51+50])
        y_train.append(words[i*51+50])

    for i in range(0,10000,1):
        temp_x = []
        temp_y = []
        # for k in range(0,50):
        #     #对输入x每个词进行向量化
        #     print(x_train[i][k])
        #     print(wv_model.wv(x_train[i][k]))
        #     temp_x.append(wv_model.wv(x_train[i][k]))
        # #对输出y进行向量化
        # temp_y.append(wv_model.wv(y_train[i]))
        try:
            #对输入x每个词进行向量化
            temp_x = wv_model.wv[x_train[i]]
            # #对输出y进行向量化
            temp_y = wv_model.wv[y_train[i]]
            X_train.append(temp_x)
            Y_train.append(temp_y)
        except KeyError:
            print()

    return X_train,Y_train

X_train,Y_train = get_train_dataset(words)

print(len(X_train))
print(len(Y_train))
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(Y_train.shape)
print(X_train.shape)


#3.3构建神经网络
# build RNN model
model = Sequential()

# RNN cell
model.add(GRU(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),  # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#3.4训练神经网络
# training
for step in range(100):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
    Y_batch = Y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
model.save('./s2vec Model')


