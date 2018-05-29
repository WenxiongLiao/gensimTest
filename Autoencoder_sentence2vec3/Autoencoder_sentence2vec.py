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
np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

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



#3


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
    X_train =[]


    size = int(size/50)-1
    j=0
    for i in range(0,size,1):
        x_train.append(words[i*50:i*50+50])
    count=0
    for i in range(0,100,1):
        temp_x = []
        try:
            #对输入x每个词进行向量化
            temp_x = wv_model.wv[x_train[i]]
            X_train.append(temp_x)
        except KeyError:
            print(++count)

    return X_train

X_train = get_train_dataset(words)

X_train = np.array(X_train)
print(X_train.shape)


X_train = X_train.reshape((X_train.shape[0], -1))

# in order to plot in a 2D figure
encoding_dim = 1

# this is our input placeholder
input_img = Input(shape=(5000,))

# encoder layers
encoded = Dense(1000, activation='relu')(input_img)
encoded = Dense(400, activation='relu')(encoded)
encoder_output = Dense(100, activation='relu')(encoded)


# decoder layers
decoded = Dense(400, activation='relu')(encoder_output)
decoded = Dense(1000, activation='relu')(decoded)
decoded = Dense(5000, activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)

# construct the encoder model for plotting
encoder = Model(input=input_img, output=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(X_train, X_train,
                nb_epoch=20,
                batch_size=256,
                shuffle=True)

X_test = X_train[0]
X_test = X_test.reshape(-1,5000)
encoded_imgs = encoder.predict(X_test)
print(encoded_imgs[0])
encoder.save('./encoded Model')
# decoded.save('./decoded Model')
# plotting
# plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=X_train)
# plt.show()