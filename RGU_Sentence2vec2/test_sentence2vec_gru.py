import tensorflow as tf
import numpy as np
import gensim as gensim


from keras.models import load_model

wv_model = gensim.models.Word2Vec.load('./word2vecModel')
rgu_model = load_model('./s2vec Model')
test_stence = ['anarchism','originated','as','a','term','of','abuse','first','used','against','early','working','class','radicals','including','the','diggers','of','the','english','revolution','and','the','sans','culottes','of','the','french','revolution','whilst','the','term','is','still','used','in','a','pejorative','way','to','describe','any','act','that','used','violent','means','to','destroy','the']
# print(len(test_stence))
#词向量化
x_test = wv_model.wv[test_stence]

#构建gru网络
x_test = tf.reshape(x_test, [-1, 50, 100])
cell = tf.nn.rnn_cell.GRUCell(100)
init_state = cell.zero_state(1, dtype=tf.float32)
output, state = tf.nn.dynamic_rnn(cell, x_test, initial_state=init_state, time_major=False)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    output,state = sess.run([output,state])

    # print(output)
    print(state)