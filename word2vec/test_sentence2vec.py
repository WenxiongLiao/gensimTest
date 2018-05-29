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
rgu_model = load_model('./s2vec Model')
test_stence = ['anarchism','originated','as','a','term','of','abuse','first','used','against','early','working','class','radicals','including','the','diggers','of','the','english','revolution','and','the','sans','culottes','of','the','french','revolution','whilst','the','term','is','still','used','in','a','pejorative','way','to','describe','any','act','that','used','violent','means','to','destroy','the']
print(len(test_stence))
x_test = wv_model.wv[test_stence]
X_test =[]
for i in range(50):
    X_test.append(x_test)
X_test = np.array(X_test)


print(rgu_model.predict(X_test,batch_size=50)[0])
w = ['any','act','that','used','violent','means','to','destroy','the','organization','of','society','it','has','also','been','taken','up','as','a','positive','label','by','self','defined','anarchists','the','word','anarchism','is','derived','from','the','greek','without','archons','ruler','chief','king','anarchism','as','a','political','philosophy','is','the','belief','that','rulers','are','unnecessary','and','should','be','abolished','although','there','are','differing','interpretations','of','what','this','means','anarchism','also','refers','to','related','social','movements','that','advocate','the','elimination','of','authoritarian','institutions','particularly','the','state','the','word','anarchy','as','most','anarchists']