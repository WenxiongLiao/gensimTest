from gensim import corpora
from collections import defaultdict
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# 去掉停用词
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# 去掉只出现一次的单词
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

print(texts)

dictionary = corpora.Dictionary(texts)   # 生成词典

print(dictionary)
print(dictionary.token2id)
print(dictionary.dfs)
# 将文档存入字典，字典有很多功能，比如
# diction.token2id 存放的是单词-id key-value对
# diction.dfs 存放的是单词的出现频率（key,frequence）
dictionary.save('tmp/deerwester.dict')  # store the dictionary, for future reference
#corpus为某个document中（key,frequence）
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)
corpora.MmCorpus.serialize('tmp/deerwester.mm', corpus)  # store to disk, for later use
#除了MmCorpus以外，还有其他的格式，例如SvmLightCorpus, BleiCorpus, LowCorpus等等，用法类似。
#相反，可以用corpus = corpora.MmCorpus('/tmp/deerwester.mm')来从磁盘中读取corpus。