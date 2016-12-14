import gensim

from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
import itertools
import gensim
from gensim import corpora, similarities, models
from methods import get_newsgroup_train, get_newsgroup_test, get_newsgroup_all, tokenize, tokenizer
from time import time
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle
import numpy as np

newsgroup_all = get_newsgroup_all();
tokenized_data = [tokenize(doc) for doc in newsgroup_all.data]
dictionary = corpora.Dictionary(tokenized_data)
#dictionary.save('questions.dict');
corpus = [dictionary.doc2bow(text) for text in tokenized_data]
corpora.MmCorpus.serialize('/home/ubuntu/newsgroup_all.mm', corpus)
mm = corpora.MmCorpus('/home/ubuntu/newsgroup_all.mm')
lda_newsgroup = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=20)
f = open('/home/ubuntu/topic_doc_2vec_document','w')

def get_topic(lda_model,id2word,doc):
	tokens = tokenize(doc)
	bow = id2word.doc2bow(tokens)
	topic_dist = lda_model.get_document_topics(bow)
	return sorted(topic_dist,key=lambda x:x[1],reverse=True)[0][0]

doc_to_topic_list = [get_topic(lda_newsgroup,dictionary,doc) for idx,doc in enumerate(newsgroup_all.data)]

def map_doc_to_topic(lda_model,id2word, doc_id,doc):
	doc_prefix = "doc_id"
	topic_prefix = "topic_id"
	doc_to_topic = [topic_prefix+"_"+str(doc_to_topic_list[doc_id]), doc_prefix+"_"+str(doc_id)]     
	return doc_to_topic



class LabeledLineSentence_20newsgroups(object):
	def __init__(self,datafile,lda_model):
		self.file = datafile
		self.sentences = []
		self.lda_model = lda_model
		self.id2word = self.lda_model.id2word
	def __iter__(self):
		for idx,line in enumerate(self.file.data):
			tags_doc = map_doc_to_topic(self.lda_model,self.id2word,idx,line)
			yield LabeledSentence(words=tokenize(line), tags=tags_doc)
	def to_array(self):
		for idx, line in enumerate(self.file.data):
			tags_doc = map_doc_to_topic(self.lda_model,self.id2word,idx,line)
			self.sentences.append(LabeledSentence(words = tokenize(line),tags = tags_doc))
		return self.sentences
	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences


n_iters = 20
t0 = time()
it = LabeledLineSentence_20newsgroups(newsgroup_all,lda_newsgroup)
model = Doc2Vec(size=400, window=10, min_count=4, dm=0,dbow_words=1,
                              workers=50, alpha=0.025, min_alpha=0.025)
model.build_vocab(it.to_array())
print("done in %0.3fs." % (time() - t0))
#done in 39.556s.
index = 0
for epoch in range(n_iters):
	t0 = time()
	model.train(it.sentences_perm())
	model.alpha -= 0.002 # decrease the learning rate
	model.min_alpha = model.alpha
	index = index + 1
	print("done in %0.3fs." % (time() - t0))
	f.write("done in %0.3fs." % (time() - t0))

f.close()
# avg_doc_vectors =np.array([model.docvecs["doc_id_"+str(i)]+model.docvecs['topic_id_'+str(doc_to_topic_list[i])] for i in range(n_docs)])
#0.3172779496243924 from 10 iterations
# avg_doc_vectors =np.array([model.docvecs['topic_id_'+str(doc_to_topic_list[i])] for i in range(n_docs)])
avg_doc_vectors =np.array([model.docvecs["doc_id_"+str(i)] for i in range(len(newsgroup_all.data))])
#0.2951833848873177
#find the labels of the newsgroup data
#find the labels of the newsgroup data
labels = newsgroup_all.target



n_iter = 10
X_train, X_test, classification_train,classification_test = train_test_split(avg_doc_vectors, labels, test_size=0.2, random_state = 0) 
cv = ShuffleSplit(X_train.shape[0],n_iter=n_iter,test_size = 0.2,random_state=0)

# for idx,item in enumerate(avg_doc_vectors):
# 	idx = idx + 1
# 	if not isinstance(item,np.ndarray):
# 		print idx
#logistic regression
logreg = LogisticRegressionCV(cv = cv, penalty = "l2",refit = True,multi_class = "multinomial")
logreg.fit(X_train,classification_train)
pred_test = logreg.predict(X_test)

def calcErrorRate(model_results,correct_results):
	error = 0
	for i,j in zip(model_results,correct_results):
		if (i!=j):
			error += 1
	return float(error)/float(len(model_results))



#0.32346442775077333