import numpy as np
import tqdm

class N_Gram:

    def __init__(self, n, vocab):
        # n = 1 -> unigram P(wt​​)
        # n = 2 -> bigram  P(wt​∣​wt−1​)
        # n = 3 -> trigram P(wt​∣wt−2​,wt−1​)

        self.n = n 

        self.vocab_size = len(vocab)

        cnts_shape = [self.vocab_size] * n 
        cnts_shape = tuple(cnts_shape)

        self.cnts = {}
 
        self.map_token_id = {
            token:id for id, token in enumerate(vocab)
        }


    def train(self, corpus_tokenized):
        
        token_ids = []
        
        print(f"Train {self.n}-Gram:")

        for token in tqdm.tqdm(corpus_tokenized, position=0, leave=True):
       
            token_id = self.map_token_id[token]

            token_ids.append(token_id)

            # Need sufficient amount of token
            # e.g. n = 3 -> need last 2 tokens!

            if len(token_ids) < self.n:
                continue
            
            else:
                
                try:
                    self.cnts[tuple(token_ids)] += 1

                except KeyError:
                    self.cnts[tuple(token_ids)] = 1

                token_ids.pop(0)

       


    def get_prob(self, token_ids):

        token_ids_tmp = token_ids.copy()

        try:
            cnt_target = self.cnts[tuple(token_ids_tmp)] + 1
        except KeyError:
            cnt_target = 1

        cnt_list = []
        for token_id in range(self.vocab_size):
            
            token_ids_tmp[-1] = token_id

            try:
                cnt = self.cnts[tuple(token_ids_tmp)] + 1
            except KeyError:
                cnt = 1

            cnt_list.append(cnt)
            
        cnts = np.sum(cnt_list)

        prob = cnt_target / cnts 

        return prob


    # add_one: false -> MLE
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


    def perplexity(self, corpus_val_tokenized):

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
                
                prob = self.get_prob(token_ids)
      
                logit = np.log(prob)
            
                logit_list.append(logit)

                token_ids.pop(0)

        tmp = - np.average(logit_list)
       
        return np.exp(tmp)