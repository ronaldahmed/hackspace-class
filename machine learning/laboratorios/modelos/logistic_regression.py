import numpy as np
import scipy.optimize.lbfgsb as opt2
from modelos.my_math_utils import *
from modelos.linear_classifier import LinearClassifier

import ipdb


class LogisticRegression(LinearClassifier):

    def __init__(self,learning_rate=1,regularizer=1):
        # Velocidad de aprendizaje
        self.learning_rate = learning_rate
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
        # Flag si esta entrenado
        self.trained = False
        self._iter_cnt = 0

        
    def train(self,x,y):
        x_orig = x.copy()
        x = self.add_intercept_term(x)
        self.N,self.M = x.shape
        classes = np.unique(y)
        self.nr_c = classes.shape[0]
        for _class in classes:
            print("Optimizando para clase %i..." % _class)
            ## Reformatear Y
            y_class = (y==_class).astype(int)

            ## Inicializacion de parametros 
            init_parameters = np.random.randn(self.M,1)
            # Opminizacion
            w_class = self.minimize_lbfgs(init_parameters,x,y_class)
            self.w.append(w_class)
        self.w = np.hstack(self.w)
        self.trained = True

        y_pred = self.test(x_orig,self.w)
        acc = self.evaluate(y,y_pred)
        print("Accuracy: %.2f" % acc)
        

    def minimize_lbfgs(self,parameters,x,y):
        parameters2 = parameters.reshape([self.M],order="F")
        # minimizador L-BFGS-B
        result,_,_ = opt2.fmin_l_bfgs_b(self.get_objective,parameters2,args=[x,y],maxiter=50)
        return result.reshape([-1,1],order="F")

    def hyphotesis(self,x,w):
        return 1.0 / (1.0 + np.exp(-np.dot(x,w)))


    def get_objective(self,w,x,y):
        #w = parameters.reshape([M,nr_c],order="F")
        w = w.reshape([self.M,1],order='F')
        N,M = x.shape

        ## Cost function: L
        h = self.hyphotesis(x,w)
        objective = np.dot(    y.T , np.log(h + 1e-10)  ) + \
                    np.dot((1-y).T , np.log(1-h + 1e-10))
        objective /= -N
        objective = objective[0][0]

        ## Gradient:  delta_L
        gradient = np.dot( x.T, h-y)
        gradient /= N

        gradient = gradient.reshape(-1,order='F')

        self._iter_cnt+=1
        if self._iter_cnt%10 == 0:
            print("Objective = %.4f" % objective)
        return objective,gradient
    
