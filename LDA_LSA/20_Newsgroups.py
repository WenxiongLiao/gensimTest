from sklearn import datasets
import re
import gensim


from gensim.parsing.preprocessing import STOPWORDS
news_dataset=datasets.fetch_20newsgroups(shuffle=True, random_state=1,remove=('headers', 'footers', 'quotes'))
documents=news_dataset.data




stopwords = STOPWORDS
#print(stopwords)

def tokenize(text):
    text = text.lower()
    words = re.sub("\W"," ",text).split()
    words = [w for w in words if w not in stopwords]
    return words

processed_docs = [tokenize(doc) for doc in documents]
#obtain: (word_id:word)
word_count_dict = gensim.corpora.Dictionary(processed_docs)

word_count_dict.filter_extremes(no_below=20)
# no more than 20% documents

bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]

tfidf_model = gensim.models.TfidfModel(bag_of_words_corpus)
corpus_tfidf = tfidf_model[bag_of_words_corpus]

lda_model = gensim.models.LdaModel(corpus_tfidf, num_topics=20, id2word=word_count_dict, passes=5)

print(lda_model.print_topics(num_topics=10, num_words=5))


