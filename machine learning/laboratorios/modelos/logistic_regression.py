import numpy as np
import scipy.optimize.lbfgsb as opt2
from modelos.my_math_utils import *
from modelos.linear_classifier import LinearClassifier

import ipdb


class LogisticRegression(LinearClassifier):

    def __init__(self,regularizer=1):
        # Hiperparametro de regularizacion
        self.regularizer = regularizer
        # Parametros
        self.w = []
        # numero de muestras
        self.N = 0
        # numero de caracteristicas por feature
        self.M = 0
        # Numero de clases
        self.nr_c = 0
        
    def train(self,x,y):
        x = self.add_intercept_term(x)
        self.N,self.M = x.shape
        classes = np.unique(y)
        self.nr_c = classes.shape[0]
        for _class in classes:
            print("Optimizando para clase %i..." % _class)
            ## Reformatear Y
            _y = (y==_class).astype(int)

            ## Inicializacion de parametros 
            init_parameters = np.zeros(self.M,dtype=float)
            # Opminizacion
            w_class = self.minimize_lbfgs(init_parameters,x,_y)
            self.w.append(w_class)
        self.w = np.hstack(self.w)

        self.trained = True
        

    def minimize_lbfgs(self,parameters,x,y):
        # minimizador L-BFGS-B
        result,_,_ = opt2.fmin_l_bfgs_b(self.get_objective,parameters,args=[x,y])
        return result.reshape([-1,1])

    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    ############
    ### Obj =  -sum_(x,y) p(y|x) + sigma*||w||_2^2
    ### Obj = \sum_(x,y) -w*x + log(\sum_(y') exp(w*x)) +  sigma*||w||_2^2
    ############
    def get_objective(self,w,x,y):
        #w = parameters.reshape([M,nr_c],order="F")
        w = w.reshape([self.M,1],order='F')
        ## Cost function
        acum_objective = np.sum( np.dot(    y.T, np.log(self.sigmoid(np.dot(x,w)))   ) + \
                                 np.dot((1-y).T, np.log(1-self.sigmoid(np.dot(x,w))) )   )

        
        objective = -(1.0/self.N)*acum_objective + 0.5*self.regularizer*l2norm_squared(w)
        
        ## Gradient
        
        #gradient = -(1.0/self.N)*self.regularizer*np.dot(x.T,(self.sigmoid(np.dot(x,w))-y)) + self.regularizer*w
        #gradient = gradient.reshape(-1,order='F')
        gradient = np.zeros(self.M)

        for j in range(self.M):
            _sum = 0
            for i in range(self.N):
                _sum += (self.sigmoid(np.dot(x[i,:],w))-y[i,0]) * x[i,j]
            gradient[j] = -(1.0/self.N)*self.regularizer*_sum + self.regularizer*w[j,0]
        

        #ipdb.set_trace()

        print("Objective = %.3f" % objective)
        return objective,gradient
    