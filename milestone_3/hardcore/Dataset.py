import numpy as np

class Dataset:

    def __init__(self, batch_size, corpus_tokenized, vocab):
        
        self.batch_size = batch_size

        self.map_token_to_id = {token:id for id, token in enumerate(vocab)}

        self.corpus_tokenID = np.array(
            [self.map_token_to_id[token] for token in corpus_tokenized]
        )

        
    
        num_tokens = self.corpus_tokenID.shape[0]
     
        # pairing
        num_cut_off = num_tokens % 2
        num_tokens = num_tokens - num_cut_off

        self.corpus_tokenID = self.corpus_tokenID[:num_tokens]

        self.corpus_tokenID = np.reshape(self.corpus_tokenID, shape=(-1, 2))
        
        # batching
        num_samples = self.corpus_tokenID.shape[0]
        num_cut_off = num_samples % batch_size
        num_samples = num_samples - num_cut_off
        
        self.corpus_tokenID = self.corpus_tokenID[:num_samples, :]

        self.corpus_tokenID = np.reshape(self.corpus_tokenID, shape=(-1, batch_size, 2))

        self.shuffle()

        
    def shuffle(self):
        self.corpus_tokenID = np.reshape(self.corpus_tokenID, shape=(-1, 2))

        num_samples = self.corpus_tokenID.shape[0]

        idxs = np.random.permutation(num_samples)

        self.corpus_tokenID = self.corpus_tokenID[idxs, :]


        self.corpus_tokenID = np.reshape(self.corpus_tokenID, shape=(-1, self.batch_size, 2))

