# Gensim使用Python标准的日志类来记录不同优先级的各种事件，想要激活日志（可选的），运行如下代码：
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
# 导入gensim并创建一个小小的语料库，其中有9篇文档和12个属性
corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
          [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
          [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
          [(0, 1.0), (4, 2.0), (7, 1.0)],
          [(3, 1.0), (5, 1.0), (6, 1.0)],
           [(9, 1.0)],
           [(9, 1.0), (10, 1.0)],
           [(9, 1.0), (10, 1.0), (11, 1.0)],
          [(8, 1.0), (10, 1.0), (11, 1.0)]]

#初始化一个转换
tfidf = models.TfidfModel(corpus)
#将文档的一种向量表示方法转换为另一种（，以便我们从特定的角度更好地分析数据）
vec = [(0, 1), (4, 1)]
print(tfidf[vec])#[(0, 0.8075244024440723), (4, 0.5898341626740045)]


#为了将整个语料库通过Tf-idf转化并索引，以便相似度查询，需要做如下准备：
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)

#为了查询我们需要的向量vec相对于其他所有文档的相似度，需要：
sims = index[tfidf[vec]]
print(list(enumerate(sims)))  #[(0, 0.4662244), (1, 0.19139354), (2, 0.24600551), (3, 0.82094586), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0)]
sims_list = list(enumerate(sims))

max = 0
for i in range(len(sims_list)):
    if sims_list[i][1]>max:
        max = sims_list[i][1]
        index = i

print('与第'+str(index+1)+'最像，相似度为：'+str(max))
