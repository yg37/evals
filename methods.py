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
from gensim.utils import smart_open, simple_preprocess
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.parsing.preprocessing import STOPWORDS
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


def get_newsgroup_train():	
	#load 20 newsgroup data
	newsgroup_train = fetch_20newsgroups(subset='train',
	                                      remove=('headers', 'footers', 'quotes'))
	return newsgroup_train

def get_newsgroup_test():
	newsgroup_test = fetch_20newsgroups(subset = 'test',remove=('headers', 'footers', 'quotes'))
	return newsgroup_test

def get_newsgroup_all():
	newsgroup_all = fetch_20newsgroups(remove = ('headers', 'footers', 'quotes'))
	return newsgroup_all

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def generateSentences(newsgroup_data):
	result = []
	for data in newsgroup_data:
		for element in data.split('.'):
			result.append(tokenize(element))
	return result

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
    
def tokenizer(document):
    text = "".join([ch for ch in document if ch not in string.punctuation])
    text_list = text.split()
    normalized_text = [x.lower() for x in text_list]
    # Define an empty list
    nostopwords_text = []
    # Scan the words
    for word in normalized_text:
        # Determine if the word is contained in the stop words list
        if word not in ENGLISH_STOP_WORDS:
            # If the word is not contained I append it
            nostopwords_text.append(word)
    tokenized_text = [word for word in nostopwords_text if re.search('[a-zA-Z]{2,}', word)]
    return tokenized_text
