import gensim as gensim

# sentences = gensim.models.word2vec.LineSentence('./text8')
#
# model =gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
#
# #model.save('./word2vecModel')
model = gensim.models.Word2Vec.load('./word2vecModel')

#comouter的vec
print(model.wv['computer'] )

#计算词的关系
print(model.wv.most_similar(positive=['woman', 'king'], negative=['man']))

print(model.wv.most_similar(positive=['boy', 'girl'], negative=['man']))

print(model.wv.similarity('woman', 'man'))
print(model.wv.similarity('boy', 'girl'))
