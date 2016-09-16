import numpy as np
from modelos.linear_classifier import LinearClassifier


class Perceptron(LinearClassifier):
    def __init__(self,nr_epochs = 10,learning_rate = 1, averaged = True):
        LinearClassifier.__init__(self)
        # Indica si el model esta entrenado o no
        self.trained = False
        # Numero de epocas
        self.nr_epochs = nr_epochs
        # Tasa/velocidad de aprendizaje
        self.learning_rate = learning_rate
        # Indica si se promediarán los parámetros al final del entrenamiento
        self.averaged = averaged
        # Guarda los parámetros por época
        self.params_per_round = []
        # Parametros entrenados
        self.w = []

    def percentron_update(self,x,y,w):
        '''
        @param x [1xM]: instancia de entrenamiento
        @param y [1x1]: etiqueta/clase de instancia
        @param w [M x num clases] : parámetros actuales del modelo
        return : w [M x num clases] : parámetros actualizados con instancia actual
        '''
        y_pred = self.get_label(x,w)
        _lambda = 0.005

        if(y != y_pred):
            #Increase features of the truth
            w[:,y]     += self.learning_rate*x.transpose() + _lambda*w[:,y]
            #Decrease features of the prediction
            w[:,y_pred] -= self.learning_rate*x.transpose() + _lambda*w[:,y_pred]

        return w

    def train(self,X,Y):
        '''
        @param X [NxM]: vector de caracteristicas (features)
            N: numero de muestras
            M: numero de caracteristicas
        @param Y [Nx1]: vector de etiquetas (clases)
        '''

        self.params_per_round = []
        x_orig = X[:,:]
        X = self.add_intercept_term(X)
        N,M = X.shape
        nr_c = np.unique(Y).shape[0]
        w = np.zeros((M,nr_c))

        ## Randomize/shuffle the examples
        perm = np.random.permutation(N)

        for epoch_nr in range(self.nr_epochs):
             for nr in range(N):
                #print "iter %i" %( epoch_nr*N + nr)
                inst = perm[nr]
                x = X[inst:inst+1,:]
                y = Y[inst:inst+1,0]

                #############################
                # Actualizacion de parámetros
                w = self.percentron_update(x,y,w)

             self.params_per_round.append(w.copy())
             self.trained = True
             Y_pred = self.test(x_orig,w)
             acc = self.evaluate(Y,Y_pred)
             self.trained = False
             print("Epoch %i | Accuracy: %f" %( epoch_nr,acc))
        self.trained = True
        
        self.w = w
        if(self.averaged == True):
            new_w = 0
            for old_w in self.params_per_round:
                new_w += old_w
            new_w = new_w / len(self.params_per_round)
            self.w = new_w
        


