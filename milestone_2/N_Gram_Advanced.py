import numpy as np

from N_Gram_Basic import *

class N_Gram_Advanced:

    def __init__(self, n, vocab):
        
        # n = 1 -> unigram P(wt​​)
        # n = 2 -> bigram  P(wt​∣​wt−1​)
        # n = 3 -> trigram P(wt​∣wt−2​,wt−1​)

        assert(1 <= n)

        self.n = n 
        self.vocab = vocab

        self.n_gram_list = [
            N_Gram_Basic(i, vocab) for i in range(1, n + 1) # last element of range is exclusive ;)
        ]
        
        # order reversed!
        # e.g. 4-gram, 3-gram, 2-gram, 1-gram
        self.weights = np.random.rand(n)
        # Normalize
        self.weights = self.weights / np.sum(self.weights)


        self.map_token_to_id = {
            token:id for id, token in enumerate(vocab)
        }


    def train(self, corpus_tokenized):
        
        for n_gram in self.n_gram_list:
            n_gram.train(corpus_tokenized)
       


    # Backoff logic
    def get_prob_backoff_logic(self, token_ids_window):

        token_ids_window_tmp = token_ids_window.copy()

        for n_gram in reversed(self.n_gram_list):

            try:
                prob = n_gram.get_prob_raiseKeyError(token_ids_window_tmp)

                return prob 
                 
            except KeyError:
                token_ids_window_tmp.pop(0)

    
    def get_prob_interpolation(self, token_ids):

        token_ids_tmp = token_ids.copy()

        prob = 0
        for i, n_gram in enumerate(reversed(self.n_gram_list)):
            
            weight = self.weights[i]
            prob += weight * n_gram.get_prob(token_ids_tmp)

            token_ids_tmp.pop(0)

        return prob
          


    # Generation based on 
    def get_distri(self, token_ids_window, get_prob):
        
        distri = np.zeros(shape=(self.vocab_size),)

        # Prevent side effect
        token_ids_window_tmp = token_ids_window.copy()

        token_ids_window_tmp.append(None) # dummy

        for token_id in range(self.vocab_size):

            token_ids_window_tmp[-1] = token_id

            prob = get_prob(token_ids_window_tmp)

            distri[token_id] = prob

        return distri


    def perplexity(self, corpus_test_tokenized, get_prob):
        
        print("Compute perplexity")

        tokens = corpus_test_tokenized[0:self.n]
        token_ids_window = [self.map_token_to_id[id] for id in tokens]

        logit_list = []
        for token in tqdm.tqdm(corpus_test_tokenized[self.n:], position=0, leave=True):

            prob = get_prob(token_ids_window)
      
            logit = np.log(prob)
            
            logit_list.append(logit)

            
            token_id = self.map_token_to_id[token]
            token_ids_window.append(token_id)

            token_ids_window.pop(0)
            
        tmp = - np.average(logit_list)
       
        return np.exp(tmp)