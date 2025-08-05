import numpy as np
import tqdm

from Softmax import *
from CategoricalCrossEntropyLoss import *

class Neural_Bigram:

    def __init__(self, vocab_size):
        
        self.vocab_size = vocab_size

        self.embedding = np.random.uniform(low=-1, high=1, size=(vocab_size, vocab_size))

        self.softmax = Softmax()

        self.cce_loss = CategoricalCrossEntropyLoss(num_classes=vocab_size)

    def forward(self, idx):

        embeddings = self.embedding[idx, :]

        out = self.softmax(embeddings)

        return out
    
    def test(self, test_ds):

        loss_list = []
        ppl_list = []

        print("Test")
        for batch in tqdm.tqdm(test_ds.corpus_tokenID, position=0, leave=True):
            x = batch[:, 0]
            targets = batch[:, 1]

            y_hat = self.forward(x)

            loss = self.cce_loss(y_hat, targets)
            loss_list.append(loss)

            ppl = np.exp(loss)
            ppl_list.append(ppl)

        
        loss = np.average(loss_list)
        ppl = np.average(ppl_list)

        return loss, ppl
    
    def save(self, file):
        np.save(file=file, arr=self.embedding)
    
    def load(self, file):
        self.embedding = np.load(file)
