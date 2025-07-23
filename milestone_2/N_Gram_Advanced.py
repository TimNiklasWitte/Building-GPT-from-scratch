import numpy as np

from N_Gram_Basic import *

class N_Gram_Advanced:

    def __init__(self, n, vocab):
        
        self.n = n 
        self.vocab = vocab

        self.n_gram_list = [
            N_Gram_Basic(i, vocab) for i in range(1, n + 1)
        ]

        self.weights = np.random.rand(n)
        # Normalize
        self.weights = self.weights / np.sum(self.weights)


        self.map_token_id = {
            token:id for id, token in enumerate(vocab)
        }


    def train(self, corpus_tokenized):
        
        for n_gram in self.n_gram_list:
            n_gram.train(corpus_tokenized)
       


    # Backoff logic
    def get_prob_backoff_logic(self, token_ids):

        token_ids_tmp = token_ids.copy()

        for n_gram in reversed(self.n_gram_list):

            try:
                prob = n_gram.get_prob_raiseKeyError(token_ids_tmp)

                return prob 
                 
            except KeyError:
                token_ids_tmp.pop(0)

    
    def get_prob_interpolation(self, token_ids):

        token_ids_tmp = token_ids.copy()

        prob = 0
        for i, n_gram in enumerate(reversed(self.n_gram_list)):
            
            weight = self.weights[i]
            prob += weight * n_gram.get_prob(token_ids_tmp)

            token_ids_tmp.pop(0)

        return prob
          


    def get_distri(self, token_ids_window):

        token_ids_window.append(None) # dummy

        token_ids = token_ids_window
    
        cnt_list = []
        for token_id in range(self.vocab_size):

            token_ids[-1] = token_id

            try:
                cnt = self.cnts[tuple(token_ids)] + 1
            except KeyError:
                cnt = 1

            cnt_list.append(cnt)

        cnts = np.array(cnt_list)
        cnts_sum = np.sum(cnts)

        distri = cnts / cnts_sum

        return distri


    def perplexity(self, corpus_val_tokenized, get_prob):

        token_ids = [] 

        logit_list = []
        for token in corpus_val_tokenized:

            token_id = self.map_token_id[token]

            token_ids.append(token_id)

            # Need sufficient amount of token
            # e.g. n = 3 -> need last 2 tokens!
            if len(token_ids) < self.n:
                continue
                
            else:
                
                prob = get_prob(token_ids)
      
                logit = np.log(prob)
            
                logit_list.append(logit)

                token_ids.pop(0)

        tmp = - np.average(logit_list)
       
        return np.exp(tmp)