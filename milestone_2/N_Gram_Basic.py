import numpy as np
import tqdm

class N_Gram_Basic:

    def __init__(self, n, vocab):
        # n = 1 -> unigram P(wt​​)
        # n = 2 -> bigram  P(wt​∣​wt−1​)
        # n = 3 -> trigram P(wt​∣wt−2​,wt−1​)

        assert(1 <= n)


        self.n = n 

        self.vocab_size = len(vocab)

        # e.g. P(wt​∣wt−2​,wt−1​) -> order: wt−2​,wt−1, wt
        self.cnts = {}
 
        self.map_token_to_id = {
            token:id for id, token in enumerate(vocab)
        }


    def train(self, corpus_tokenized):
        
        tokens = corpus_tokenized[0:self.n]
        token_ids_window = [self.map_token_to_id[id] for id in tokens]
        
        print(f"Train {self.n}-Gram:")

        for token in tqdm.tqdm(corpus_tokenized[self.n:], position=0, leave=True):
       
            try:
                self.cnts[tuple(token_ids_window)] += 1

            except KeyError:
                self.cnts[tuple(token_ids_window)] = 1
            
            token_id = self.map_token_to_id[token]
            token_ids_window.append(token_id)

            token_ids_window.pop(0)

       


    def get_prob(self, token_ids_window):

        #
        # Laplace smoothing
        #

        try:
            cnt_target = self.cnts[tuple(token_ids_window)] + 1
        except KeyError:
            cnt_target = 1

        # Prevent side effect
        token_ids_window_tmp = token_ids_window.copy()

        cnt_list = []
        for token_id in range(self.vocab_size):
            
            token_ids_window_tmp[-1] = token_id

            try:
                cnt = self.cnts[tuple(token_ids_window_tmp)] + 1
            except KeyError:
                cnt = 1

            cnt_list.append(cnt)
            
        cnts = np.sum(cnt_list)

        prob = cnt_target / cnts 

        return prob
    

    # No Laplace-smoothing -> trigger KeyError (later used for backoff)
    def get_prob_raiseKeyError(self, token_ids_window):

        try:
            cnt_target = self.cnts[tuple(token_ids_window)]
        except KeyError:
            raise KeyError
        
        # Prevent side effect
        token_ids_window_tmp = token_ids_window.copy()

        cnt_list = []
        for token_id in range(self.vocab_size):
            
            token_ids_window_tmp[-1] = token_id

            try:
                cnt = self.cnts[tuple(token_ids_window_tmp)]
            except KeyError:
                cnt = 0

            cnt_list.append(cnt)
            
        cnts = np.sum(cnt_list)

        prob = cnt_target / cnts 

        return prob


    def get_distri(self, token_ids_window):

        # Prevent side effect
        token_ids_window_tmp = token_ids_window.copy()

        token_ids_window_tmp.append(None) # dummy

        cnt_list = []
        for token_id in range(self.vocab_size):

            token_ids_window_tmp[-1] = token_id

            try:
                cnt = self.cnts[tuple(token_ids_window_tmp)] + 1
            except KeyError:
                cnt = 1

            cnt_list.append(cnt)

        cnts = np.array(cnt_list)
        cnts_sum = np.sum(cnts)

        distri = cnts / cnts_sum

        return distri


    def perplexity(self, corpus_val_tokenized):
        
        print("Compute perplexity")

        tokens = corpus_val_tokenized[0:self.n]
        token_ids_window = [self.map_token_to_id[id] for id in tokens]

        logit_list = []
        for token in tqdm.tqdm(corpus_val_tokenized[self.n:], position=0, leave=True):

            prob = self.get_prob(token_ids_window)
      
            logit = np.log(prob)
            
            logit_list.append(logit)

            
            token_id = self.map_token_to_id[token]
            token_ids_window.append(token_id)

            token_ids_window.pop(0)

        tmp = - np.average(logit_list)
       
        return np.exp(tmp)