

#experiment with simple_wiki
from gensim.utils import smart_open, simple_preprocess
from gensim import corpora
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
import itertools
import os, logging, gensim, bz2
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from methods import get_newsgroup_train, get_newsgroup_test, get_newsgroup_all, tokenize, tokenizer


newsgroup_train = get_newsgroup_train()
newsgroup_test = get_newsgroup_test()
newsgroup_all = get_newsgroup_all()
n_docs = newsgroup_all.filenames.shape[0]
n_docs

def tokenize(text):
    return [token for token in simple_preprocess(text)]
doc_stream = [tokenize(data) for data in newsgroup_all.data]   
dictionary= gensim.corpora.Dictionary(doc_stream)
corpus = [dictionary.doc2bow(text) for text in doc_stream]
corpora.MmCorpus.serialize('/home/ubuntu/newsgroup.mm', corpus)
mm = corpora.MmCorpus('/home/ubuntu/newsgroup.mm')


lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, update_every=1, passes=1)

#y_pred = [sorted(lda.get_document_topics(tokenized),key=lambda x:x[1],reverse=True)[0][0] for tokenized in corpus]

labels = newsgroup_all.target



