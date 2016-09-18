import numpy as np

class Multinomial():

    def __init__(self,params):
        self.params = params

    def sample(self,points):
        return np.random.multinomial(points,self.params,size=points)
    
def estimate_multinomial(X,y):
    classes = np.unique(y)
    nr_c = classes.shape[0]
    params = np.zeros((nr_c))
    for i in xrange(nr_c):
        idx,_ = np.nonzero(y == classes[i])
        params[i] = np.mean(X[idx])
    return Multinomial(params)
