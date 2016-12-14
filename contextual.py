import gensim

from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
import itertools
import gensim
from gensim import corpora, similarities, models
from methods import get_newsgroup_train, get_newsgroup_test, get_newsgroup_all, tokenize, tokenizer,generateSentences,randomizeAndTrain
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy as np
from time import time
from random import shuffle
from gensim.models import Word2Vec
import random 
from t_SNE import visualize

ratings_url = "/Users/YaqiGuo/Downloads/SCWS/ratings.txt"

ratings = open(ratings_url,'r')
lines = [line for line in ratings]
words_1 = [line.split('\t')[1] for line in lines]
POS_1 = [line.split('\t')[2] for line in lines]
words_2 = [line.split('\t')[3] for line in lines]
POS_2 = [line.split('\t')[4] for line in lines]
contexts_1 = [line.split('\t')[5] for line in lines]
contexts_2 = [line.split('\t')[6] for line in lines]
contexts_all = contexts_1 + contexts_2
avg_ratings = [float(line.split('\t')[9]) for line in lines]

def tokenize(text):
    return [token for token in simple_preprocess(text)]
tokenized_data = [tokenize(doc) for doc in contexts_all]
dictionary = corpora.Dictionary(tokenized_data)
#dictionary.save('questions.dict');
corpus = [dictionary.doc2bow(text) for text in tokenized_data]
corpora.MmCorpus.serialize('newsgroup_all.mm', corpus)
mm = corpora.MmCorpus('newsgroup_all.mm')
ntopics = 30
lda_contextual = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=ntopics)

def get_topic(lda_model,bow):
	topic_dist = lda_model.get_document_topics(bow)
	return sorted(topic_dist,key=lambda x:x[1],reverse=True)[0][0]

doc_to_topic_list = [get_topic(lda_contextual,bow) for bow in corpus]

def map_doc_to_topic(lda_model,id2word, doc_id,topic_list):
	doc_prefix = "doc_id"
	topic_prefix = "topic_id"
	doc_to_topic = [topic_prefix+"_"+str(topic_list[doc_id]), doc_prefix+"_"+str(doc_id)]     
	return doc_to_topic


class LabeledLineSentence_contextual(object):
	def __init__(self,tokenized,lda_model):
		self.sentences=[]
		self.tokenized_docs = tokenized
		self.lda_model = lda_model
		self.id2word = self.lda_model.id2word
	def __iter__(self):
		for idx,doc in enumerate(self.tokenized_docs):
			tags_doc = map_doc_to_topic(self.lda_model,self.id2word,idx,doc_to_topic_list)
			yield LabeledSentence(words=doc, tags=tags_doc)
	def to_array(self):
		for idx, doc in enumerate(self.tokenized_docs):
			tags_doc = map_doc_to_topic(self.lda_model,self.id2word,idx,doc_to_topic_list)
			self.sentences.append(LabeledSentence(words = doc,tags = tags_doc))
		return self.sentences
	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences

class LabeledLineSentence_contextual_doc2vec(object):
	def __init__(self,tokenized):
		self.sentences=[]
		self.tokenized_docs = tokenized
	def __iter__(self):
		for idx,doc in enumerate(self.tokenized_docs):
			tags_doc = [doc_prefix+"_"+str(idx)]
			yield LabeledSentence(words=doc, tags=tags_doc)
	def to_array(self):
		for idx, doc in enumerate(self.tokenized_docs):
			tags_doc = ["doc_id"+"_"+str(idx)]
			self.sentences.append(LabeledSentence(words = doc,tags = tags_doc))
		return self.sentences
	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences


#train the doc2vec model
n_iters = 10
t0 = time()
it = LabeledLineSentence_contextual_doc2vec(tokenized_data)
doc2vec_model= Doc2Vec(size=400, window=5, min_count=4, dm=0,dbow_words=1,
                              workers=50, alpha=0.025, min_alpha=0.025)
doc2vec_model.build_vocab(it.to_array())
print("done in %0.3fs." % (time() - t0))
#done in 4.818s.
index = 0
for epoch in range(n_iters):
	t0 = time()
	doc2vec_model.train(it.sentences_perm())
	doc2vec_model.alpha -= 0.002 # decrease the learning rate
	doc2vec_model.min_alpha = model.alpha
	index = index + 1
	print("done in %0.3fs." % (time() - t0))
	f.write("done in %0.3fs." % (time() - t0))


#train the topic doc2vec model
n_iters = 20
dim = 400
t0 = time()
it = LabeledLineSentence_contextual(tokenized_data,lda_contextual)
model = Doc2Vec(size=dim, window=10, min_count=1, dm=0,dbow_words=1,
                              workers=50, alpha=0.025, min_alpha=0.025)
model.build_vocab(it.to_array())
print("done in %0.3fs." % (time() - t0))
#done in 2.018s.

f = open('/Users/YaqiGuo/Desktop/contextual_2','w')
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

f = open('word2vec_document_contextual','w')
sentences = generateSentences(contexts_all)
word2vec_model = Word2Vec(sentences, size=dim, window=2, hs = 1, workers = 5, min_count = 1)
word2vec_model = randomizeAndTrain(f,tokenized_data,word2vec_model,10)
f.close()




###########Euclidean Distance#####################
#word vectors from doc2vec model
vectors_words_1 = []
#sum of word vectors and document vectors from doc2vec model
vectors_combined_1 = []
words_1_missing = []
for idx,word in enumerate(words_1):
	if word.lower() in model.vocab:
		# vectors_1.append(word2vec_model[word.lower()])
		vectors_words_1.append(model[word.lower()])
		vectors_combined_1.append(model[word.lower()]+model.docvecs[idx])
	else:
		vectors_1.append(np.zeros(dim))
		vectors_words_1.append(np.zeros(dim))
		vectors_combined_1.append(np.zeros(dim))
		words_1_missing.append(idx)


length_of_1 = 2003
vectors_words_2 = []
vectors_combined_2 = []
words_2_missing = []
for idx,word in enumerate(words_2):
	if word.lower() in model.vocab:
		# vectors_2.append(word2vec_model[word.lower()])
		vectors_words_2.append(model[word.lower()])
		vectors_combined_2.append(model[word.lower()] + model.docvecs[idx+length_of_1])
	else:
		vectors_2.append(np.zeros(dim))
		vectors_words_2.append(np.zeros(dim))
		vectors_combined_2.append(np.zeros(dim))
		words_2_missing.append(idx)

#vectors from word2vec
vectors_word2vec_1 = []
vectors_word2vec_2 = []
for idx,word in enumerate(words_1):
	if word.lower() in word2vec_model.vocab:
		vectors_word2vec_1.append(word2vec_model[word.lower()])
	else:
		vectors_word2vec_1.append(np.zeros(dim))
for idx,word in enumerate(words_2):
	if word.lower() in word2vec_model.vocab:
		vectors_word2vec_2.append(word2vec_model[word.lower()])
	else:
		vectors_word2vec_2.append(np.zeros(dim))

from scipy.spatial import distance

distance_word = [distance.euclidean(a,b) for a,b in zip(vectors_words_1,vectors_words_2)]
distance_combined = [distance.euclidean(a,b) for a,b in zip(vectors_combined_1,vectors_combined_2)]
distance_word2vec = [distance.euclidean(a,b) for a,b in zip(vectors_word2vec_1,vectors_word2vec_2)]
from scipy.stats.stats import pearsonr
cl_word = pearsonr(distance_word, avg_ratings)
#(-0.30412581152134277, 3.9377722707767644e-44)
cl_combined = pearsonr(distance_combined,avg_ratings)
#(-0.12957839723937328, 5.8777476718156019e-09)
cl_word2vec = pearsonr(distance_word2vec,avg_ratings) 
#(-0.34146071190521737, 7.0635236495147335e-56)


###########Cosine Similarity#####################
distance_cosine_topic2vec_words = []
for a,b in zip(words_1, words_2):
	if a.lower() in model.vocab and b.lower() in model.vocab:
		distance_cosine_doc2vec_words.append(model.similarity(a.lower(),b.lower()))
	else:
		distance_cosine_doc2vec_words.append(0)

distance_cosine_doc2vec_words = []
for a,b in zip(words_1, words_2):
	if a.lower() in doc2vec_model.vocab and b.lower() in doc2vec_model.vocab:
		distance_cosine_doc2vec_words.append(doc2vec_model.similarity(a.lower(),b.lower()))
	else:
		distance_cosine_doc2vec_words.append(0)
distance_cosine_word2vec = []
for a,b in zip(words_1, words_2):
	if a.lower() in word2vec_model.vocab and b.lower() in word2vec_model.vocab:
		distance_cosine_word2vec.append(word2vec_model.similarity(a.lower(),b.lower()))
	else:
		distance_cosine_word2vec.append(0)

distance_cosine_doc2vec_words = [model.similarity(a.lower(),b.lower()) for a,b in zip(words_1,words_2)]
distance_cosine_word2vec= [word2vec_model.similarity(a.lower(),b.lower()) for a,b in zip(words_1,words_2)]


pearsonr(distance_cosine_doc2vec_words,avg_ratings)
#correlation=0.23295170632250459, pvalue=4.3346596012384097e-26)

pearsonr(distance_cosine_word2vec,avg_ratings)
#(0.34718305681323314, 7.9300262780344945e-58)


#visualization
#visualize(vectors_words_1,words_1,2)
#visualize(vectors_combined, words_combined,np.repeat((1,2),len(words_combined)/2),2)
visualize(vectors_words_1[1:11]+vectors_words_2[1:11],words_1[1:11]+words_2[1:11],np.repeat(range(1,11),2),2)
