import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
 

def visualize(vectors,labels,c,n_dim):
	tsne_model = TSNE(n_components=n_dim,random_state=0)
	np.set_printoptions(suppress=True)
	result = tsne_model.fit_transform(vectors)
	plt.scatter(result[:,0],result[:,1],c=c)
	counter = 0
	for x,y in zip(result[:,0],result[:,1]):
		plt.annotate(labels[counter],xy=(x,y))
		counter = counter + 1;
	plt.show()

    # for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
    #     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')


