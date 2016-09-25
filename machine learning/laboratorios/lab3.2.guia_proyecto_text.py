"""
Prueba de Proyecto Parcial. Texto
"""
  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import warnings
import glob as gb

import ipdb
warnings.filterwarnings("ignore")
###########################################################

## 1) Leer dataset
x_dataset = []
Y = []
for filename in gb.glob('datasets/texto/*.txt'):
	doc = []
	for line in open(filename,encoding="latin-1"):
		line = line.strip('\n')
		if line!='':
			word,lemma,pos,wordnet_id = line.split(' ')
			doc.append( word )
	x_dataset.append( doc )
	Y.append(filename)
	

## 2) preprocess
import nltk
stopword = nltk.corpus.stopwords.words('spanish')

# armar vocabulario
vocabulary = set()
for doc in x_dataset:
	for word in doc:
		if word not in stopword:
			vocabulary.add(word)
vocabulary = list(vocabulary)
V = len(vocabulary)

ipdb.set_trace()

# vectorizar documentos
X = []
for doc in x_dataset:
	x_i = np.zeros(V)
	for word in doc:
		#  1: palabra presente
		word_id = vocabulary.index(word)
		x_i[word_id] = 1.0
	X.append( x_i )
ipdb.set_trace()


## 3) Dividir en training, test
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=42)









