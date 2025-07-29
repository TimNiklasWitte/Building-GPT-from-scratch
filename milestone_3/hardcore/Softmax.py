import numpy as np

class Softmax:
    
    def __init__(self):
        pass

    def __call__(self, x):
        
        tmp = np.exp(x)

        tmp_sum = np.sum(tmp, axis=1)

        # add dummy dim for broadcast
        tmp_sum = np.expand_dims(tmp_sum, axis=1) 

        return tmp / tmp_sum