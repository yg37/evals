from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from time import time
import string
import re
# numpy
import numpy as np
import pickle
# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from methods import get_newsgroup_train, get_newsgroup_test, get_newsgroup_all, tokenize, tokenizer

#load 20 newsgroup data
newsgroup_train = get_newsgroup_train()
newsgroup_test = get_newsgroup_test()
newsgroup_all = get_newsgroup_all()
n_docs = newsgroup_all.filenames.shape[0]
n_docs



#Doc2Vec
class LabeledLineSentence_20newsgroups(object):
	def __init__(self,datafile):
		self.file = datafile
		self.sentences = []
	def __iter__(self):
		for idx,line in enumerate(self.file.data):
			yield LabeledSentence(words=line.split(), tags=["doc" + str(idx)])
			self.id = self.id + 1
	def to_array(self):
		for idx, line in enumerate(self.file.data):
			self.sentences.append(LabeledSentence(words = tokenize(line),tags = ["doc" + str(idx)]))
		return self.sentences
	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences

f = open('/home/ubuntu/eval_doc2vec_record','w')

#train the model
n_iters = 10
t0 = time()
it = LabeledLineSentence_20newsgroups(newsgroup_all)
model = Doc2Vec(size=400, window=10, min_count=4, dm=0,dbow_words=1,
                              workers=50, alpha=0.025, min_alpha=0.025)
model.build_vocab(it.to_array())
print("done in %0.3fs." % (time() - t0))
#done in 4.818s.

index = 0
for epoch in range(n_iters):
	t0 = time()
	model.train(it.sentences_perm())
	model.alpha -= 0.002 # decrease the learning rate
	model.min_alpha = model.alpha
	index = index + 1
	print("done in %0.3fs." % (time() - t0))
	f.write("done in %0.3fs." % (time() - t0))
#done in 296.992s.
#done in 295.710s.
#done in 295.071s.
#done in 294.011s.
#done in 278.883s.
#done in 278.577s.
#done in 278.520s.
#done in 278.361s.
#done in 278.285s.
#done in 278.016s.
#model.save('/Users/YaqiGuo/Desktop/2vec_tests/doc2vec')
model.save('doc2vec_model')
###########################################################
#https://github.com/RaRe-Technologies/gensim/issues/402
 f.close()

#get all the trained document vectors
avg_doc_vectors =np.array([model.docvecs["doc"+str(i)] for i in range(n_docs)])
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


#0.3044631020768891

