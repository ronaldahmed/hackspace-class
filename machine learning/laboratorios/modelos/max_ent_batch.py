import numpy as np
import scipy.optimize.lbfgsb as opt2
from modelos.my_math_utils import *
from modelos.linear_classifier import LinearClassifier
import ipdb

################
### Train a maxent in a batch setting using LBFGS
################
class MaxEnt_batch(LinearClassifier):

    def __init__(self,regularizer=1):
        self.w = []
        self.regularizer = regularizer
        
    def train(self,x,y):
        x = self.add_intercept_term(x)
        N,M = x.shape
        classes = np.unique(y)
        nr_c = classes.shape[0]
        ## Inicializar parametros
        init_parameters = np.zeros((M,nr_c),dtype=float)
        ## Inicializar matriz de conteo para E[x]
        emp_counts = np.zeros((M,nr_c))
        classes_idx = []
        for c,c_i in enumerate(classes):
            # indices de muestras con clase c
            idx,_ = np.nonzero(y == c)
            classes_idx.append(idx)
            emp_counts[:,c_i] = x[idx,:].sum(0)
        params = self.minimize_lbfgs(init_parameters,x,y,emp_counts,classes_idx,N,M,nr_c)
        self.w = params
        self.trained = True
        

    def minimize_lbfgs(self,parameters,x,y,emp_counts,classes_idx,N,M,nr_c):
        parameters2 = parameters.reshape([M*nr_c],order="F")
        result,_,d = opt2.fmin_l_bfgs_b(self.get_objective,parameters2,args=[x,y,emp_counts,classes_idx,N,M,nr_c])
        return result.reshape([M,nr_c],order="F")

    ############
    ### Obj =  -sum_(x,y) p(y|x) + sigma*||w||_2^2
    ### Obj = \sum_(x,y) -w*x + log(\sum_(y') exp(w*x)) +  sigma*||w||_2^2
    ############
    def get_objective(self,parameters,x,y,emp_counts,classes_idx,N,M,nr_c):
        w = parameters.reshape([M,nr_c],order="F")

        ## Ingrese codigo aqui
        # Funcion costo
        objective = 
        
        # Gradiente
        gradient = 

        print("Objective = %.3f" % objective)
        return objective,gradient.reshape([M*nr_c],order="F")
        
        