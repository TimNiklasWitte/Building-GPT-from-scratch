import numpy as np

class CategoricalCrossEntropyLoss:

    def __init__(self, num_classes):
        self.num_classes = num_classes 

    def __call__(self, preds, targets):
        
        loss = - np.average(np.log(preds[np.arange(len(targets)), targets]))

        return loss


