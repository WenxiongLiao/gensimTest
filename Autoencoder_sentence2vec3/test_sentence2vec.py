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
from keras.models import load_model

wv_model = gensim.models.Word2Vec.load('./word2vecModel')
encoded_model = load_model('./encoded Model')
test_stence = ['anarchism','originated','as','a','term','of','abuse','first','used','against','early','working','class','radicals','including','the','diggers','of','the','english','revolution','and','the','sans','culottes','of','the','french','revolution','whilst','the','term','is','still','used','in','a','pejorative','way','to','describe','any','act','that','used','violent','means','to','destroy','the']
# print(len(test_stence))
x_test = wv_model.wv[test_stence]

x_test = x_test.reshape(-1,5000)

print(encoded_model.predict(x_test)[0])

# X_test = X_train[0]
# X_test = X_test.reshape(-1,5000)
# encoded_imgs = encoder.predict(X_test)
# print(encoded_imgs[0])