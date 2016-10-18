import numpy as np
import scipy as scipy
import modelos.linear_classifier as lc
import sys
from distribuciones.gaussian import *
from distribuciones.multinomial import *
import ipdb


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self,smooth=0):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth_param = smooth
        
    def train(self,x,y):
        # n_docs: nro. de documentos
        # n_words: nro. of palabras unicas
        n_docs,n_words = x.shape
        
        # classes: lista de clases
        classes = np.unique(y)
        # n_classes = nro. de clases
        n_classes = classes.shape[0]
        
        # Inicializacion del Prior y variables de Likelihood
        #  P(Y)
        prior = np.zeros(n_classes)
        #  P(w_j|y)
        likelihood = np.zeros((n_words,n_classes))


        ###########################
        for c in range(n_classes):
            docs_in_class,_ = np.nonzero(y==c)
            prior[c] = 1.0*len(docs_in_class) / n_docs # P(y=c)
            
            words_in_class = x[docs_in_class].sum(0) # sum de freq en docs de clase c
            total_words_in_class = x[docs_in_class].sum() # total de freq en docs de clase c

            likelihood[:,c] = (self.smooth_param + words_in_class) / \
                              (self.smooth_param*n_words + total_words_in_class)
            

        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes):
            params[0,i] = np.log(prior[i])
            params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params











