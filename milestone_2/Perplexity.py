def perplexity(self, n_grams, corpus_val_tokenized, weights):

        token_ids = [] 

        for token in corpus_val_tokenized:

            token_id = self.map_token_id[token]

            token_ids.append(token_id)

            # Need sufficient amount of token
            # e.g. n = 4 -> need last 3 tokens!
            if len(token_ids) != self.n - 1 or self.n == 0:
                continue
                
            else:
                
                try:
                    prob = self.probs[tuple(token_ids)]
                except KeyError:

                print(prob)

                token_ids.pop(0)