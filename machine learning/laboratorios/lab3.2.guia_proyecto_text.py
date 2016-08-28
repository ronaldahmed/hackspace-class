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
# armar vocabulario

# vectorizar documentos
X = []
for doc in x_dataset:
	
	x_i = np.zeros(len(vocabulario))
	for word in doc:
		#  1: palabra presente
	X.append( x_i )


## 3) Dividir en training, test
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=42)
