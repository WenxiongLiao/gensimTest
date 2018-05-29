import os
from gensim import corpora, models, similarities
from pprint import pprint
from matplotlib import pyplot as plt
import logging

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def PrintDictionary(dictionary):
    token2id = dictionary.token2id
    dfs = dictionary.dfs
    token_info = {}
    for word in token2id:
        token_info[word] = dict(
            word = word,
            id = token2id[word],
            freq = dfs[token2id[word]]
        )
    token_items = token_info.values()
    token_items = sorted(token_items, key = lambda x:x['id'])
    print('The info of dictionary: ')
    pprint(token_items)
    print('--------------------------\n')

def Show2dCorpora(corpus):
    nodes = list(corpus)
    ax0 = [x[0][1] for x in nodes] # 绘制各个doc代表的点
    ax1 = [x[1][1] for x in nodes]
    # print(ax0)
    # print(ax1)
    plt.plot(ax0,ax1,'o')
    plt.show()

if (os.path.exists("tmp/deerwester.dict")):
    dictionary = corpora.Dictionary.load('tmp/deerwester.dict')
    corpus = corpora.MmCorpus('tmp/deerwester.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")

PrintDictionary(dictionary)

# 尝试将corpus(bow形式) 转化成tf-idf形式
tfidf_model = models.TfidfModel(corpus) # step 1 -- initialize a model 将文档由按照词频表示 转变为按照tf-idf格式表示
# doc_bow = [(0, 1), (1, 1),[4,3]]
# doc_tfidf = tfidf_model[doc_bow]
# print(doc_tfidf)

# 将整个corpus转为tf-idf格式
corpus_tfidf = tfidf_model[corpus]
# pprint(list(corpus_tfidf))
# pprint(list(corpus))

## LSI模型 **************************************************
# 转化为lsi模型, 可用作聚类或分类
lsi_model = models.lsimodel.LsiModel(corpus, id2word=dictionary, num_topics=2)
corpus_lsi = lsi_model[corpus_tfidf]
nodes = list(corpus_lsi)
print('The info of corpus_lsi: ')
print(lsi_model.print_topics(num_topics=2, num_words=5)) # 打印各topic的含义[(0, '0.040*"system" + 0.039*"interface" + 0.038*"graph"'), (1, '0.046*"survey" + 0.036*"time" + 0.036*"response"')]
pprint(nodes)
print('--------------------------\n')



# ax0 = [x[0][1] for x in nodes] # 绘制各个doc代表的点
# ax1 = [x[1][1] for x in nodes]
# print(ax0)
# print(ax1)
# plt.plot(ax0,ax1,'o')
# plt.show()

lsi_model.save('tmp/model.lsi') # same for tfidf, lda, ...
lsi_model = models.LsiModel.load('tmp/model.lsi')
#  *********************************************************

## LDA模型 **************************************************
# lda_model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=5)
lda_model = models.ldamodel.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lda = lda_model[corpus_tfidf]
node = list(corpus_lda)
print('The info of corpus_lda: ')
pprint(node)

print(lda_model.print_topics(num_topics=2, num_words=5))
Show2dCorpora(corpus_lsi)
print('--------------------------\n')

# nodes = list(corpus_lda)
# pprint(list(corpus_lda))

# 此外，还有Random Projections, Hierarchical Dirichlet Process等模型


corpus_simi_matrix = similarities.MatrixSimilarity(corpus_lda)
# 计算一个新的文本与既有文本的相关度
test_text = "Human computer interaction".split()
test_bow = dictionary.doc2bow(test_text)
test_tfidf = tfidf_model[test_bow]
test_lsi = lsi_model[test_tfidf]
print(test_lsi)

test_simi = corpus_simi_matrix[test_lsi]
print(list(enumerate(test_simi)))
sims_list = list(enumerate(test_simi))


max = 0
index = 0
for i in range(len(sims_list)):
    if sims_list[i][1]>max:
        max = sims_list[i][1]
        index = i

print('与第'+str(index+1)+'最像，相似度为：'+str(max))

print(corpus[index])
for worf_fre in corpus[index]:
    print(dictionary[worf_fre[0]],end=' ')