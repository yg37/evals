import os
from gensim.models import Word2Vec
from stemming.porter2 import stem
from autocorrect import spell
import random 
import numpy as np
import logging
import time
from time import time
from methods import get_newsgroup_train, get_newsgroup_test, get_newsgroup_all, tokenize, generateSentences, tokenizer


##TODO:
##ADD TOKENIZER!!!!!!!!!!!!!!!!
newsgroup_all = get_newsgroup_all();
tokenized_data = [tokenize(doc) for doc in newsgroup_all.data]



#done in 34.014s.
#done in 34.815s.
#done in 34.730s.
#done in 34.722s.
#done in 34.732s.
#done in 34.263s.
#done in 34.336s.
#done in 34.843s.
#done in 34.507s.
#done in 34.504s.
#done in 34.692s.
f = open('word2vec_document','w')

def randomizeAndTrain(file,sentences,model,n_iter):
	t0 = time()
	model.train(sentences)
	print("done in %0.3fs." % (time() - t0))
	for i in range(1, n_iter+1):
	    random.shuffle(sentences)
	    t0 = time()
	    model.train(sentences)
	    print("done in %0.3fs." % (time() - t0))
	    f.write("done in %0.3fs." % (time() - t0))
	return model

sentences = generateSentences(newsgroup_all.data)
model = Word2Vec(sentences, size=400, window=10, hs = 1, workers = 5, min_count = 5)
model = randomizeAndTrain(f,sentences,model,10)

def avg_sentence_vectors(model,sentence):
	vectors = np.array([model[word] for word in sentence.lower().split() if word in model.vocab.keys()])
	return vectors.mean(0)

def avg_doc_vectors(model,doc):
	#vectors = np.array(result)
	vectors = np.array([avg_sentence_vectors(model,sentence) for sentence in doc.split('.') if isinstance(avg_sentence_vectors(model,sentence),np.ndarray)])
	return vectors.mean(0)

	# return result

filtered_data = [doc for idx,doc in enumerate(newsgroup_all.data) if newsgroup_all.data[idx] != '']
filtered_data_indices = [idx for idx,doc in enumerate(newsgroup_all.data) if newsgroup_all.data[idx] != '']
doc_vectors = np.array([avg_doc_vectors(model,doc) for doc in filtered_data])
labels = [value for idx,value in enumerate(newsgroup_all.target) if idx in filtered_data_indices]


#sentences with "\n" results in nan values
filtered_indices = [idx for idx,value in enumerate(doc_vectors) if not isinstance(value,np.float64)]
filtered_doc_vectors = np.array([value for value in doc_vectors if not isinstance(value,np.float64)])
filtered_labels = np.array([label for idx,label in enumerate(labels) if idx in filtered_indices])

n_iter = 10
X_train, X_test, classification_train,classification_test = train_test_split(filtered_doc_vectors, filtered_labels, test_size=0.2, random_state = 0) 
cv = ShuffleSplit(X_train.shape[0],n_iter=n_iter,test_size = 0.2,random_state=0)

# for idx,item in enumerate(doc_vectors):
# 	idx = idx + 1
# 	if not isinstance(item,np.ndarray):
# 		print idx
# 		print type(item)
#logistic regression
logreg = LogisticRegressionCV(cv = cv, penalty = "l2",refit = True,multi_class = "multinomial")
logreg.fit(X_train,classification_train)
pred_test = logreg.predict(X_test)
f.close()





