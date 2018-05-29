from gensim import corpora,models,similarities
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

documents = ["Shipment of gold damaged in fire",
             "Delivery of silver arrived in a silver truck",
             "Shipment of go;d arrived in a truck"]

# 将英文单词小写化
texts = [[word for word in document.lower().split()]
         for document in documents]
print(texts)
# 通过这些文档抽取一个‘词袋’，将文档的token映射为id
dictionary = corpora.Dictionary(texts)
# print(dictionary)
# print(dictionary.token2id)
# 将用字符串表示的文档转换为用id表示的文档向量：
corpus = [dictionary.doc2bow(text) for text in texts]
# print(corpus)
# 例如（9，2）这个元素代表第二篇文档中id为9的单词“silver”出现了2次。

# 基于这些“训练文档”计算一个TF-IDF“模型”：
tfidf = models.TfidfModel(corpus)

# 基于这个TF-IDF模型，将上述用词频表示文档向量表示为一个用tf-idf值表示的文档向量：
corpus_tfidf = tfidf[corpus]
# for doc in corpus_tfidf:
    # print(doc)

# 训练一个LSI模型,设置topic数为2
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
lsi.print_topics(2)

# 有了这个lsi模型，我们就可以将文档映射到一个二维的topic空间中：
corpus_lsi = lsi[corpus_tfidf]
# for doc in corpus_lsi:
#     print(doc)

# LDA模型
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
lda.print_topics(2)

# 建索引
index = similarities.MatrixSimilarity(lsi[corpus])
query = "gold silver truck"
query_bow = dictionary.doc2bow(query.lower().split())
# print(query_bow)

# 再用之前训练好的LSI模型将其映射到二维的topic空间：
query_lsi = lsi[query_bow]
print(query_lsi)

# 计算其和index中doc的余弦相似度
sims = index[query_lsi]
print(list(enumerate(sims)))

# 按相似度进行排序
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sort_sims)

