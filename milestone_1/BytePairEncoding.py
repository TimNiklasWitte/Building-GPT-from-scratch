import re

import numpy as np

class BytePairEncoding:
    def __init__(self, num_types):
        self.num_types = num_types

        self.vocab = []

        # cover all printable ascci letters
        # see ascii -d
        for i in range(0, 128):
            char = chr(i)

            # skip upper chars -> word normalization: lower
            if char.isupper():
                continue

            if char.isprintable():
                self.vocab.append(char)

        self.vocab.append("\n")

    def train(self, corpus):
        
        lines = corpus.split("\n")
        
        map_word_cnt = {}
        map_word_tokens = {}
        

        for line in lines:

            line = line.split(" ")
            
            for word in line:
                word = self.normalize_word(word)

                word = word + ' '

                try:
                    map_word_cnt[word] += 1 
                except KeyError:
                    map_word_cnt[word] = 1

                    tokens = list(word)
                    map_word_tokens[word] = tokens

                
        for i in range(self.num_types):

            #
            # Most frequent pair of adjacent tokens in corpus
            #    
         
            pair_counts = {}
            for word, tokens in map_word_tokens.items():
                
                word_cnt = map_word_cnt[word]

                for token_idx in range(len(tokens) - 1):

                    pair = (tokens[token_idx], tokens[token_idx + 1])

                    try:
                        pair_counts[pair] += word_cnt
                    except KeyError:
                        pair_counts[pair] = word_cnt

            # find max
            cnt_max = 0
            pair_max = None
            for pair, cnt in pair_counts.items():
                if cnt_max < cnt:
                    cnt_max = cnt 
                    pair_max = pair 

            token_l = pair_max[0]
            token_r = pair_max[1]

            #
            # Make new token by concatenating them
            #

            token_new = token_l + token_r

            #
            # Update vocabulary
            #

            self.vocab.append(token_new)

            #
            # Replace each occurance with token_l and token_r with token_new
            #

        
            for word, tokens in map_word_tokens.items():

                tokens_new = []

                token_idx = 0

                while token_idx < len(tokens):
                    
                    token = tokens[token_idx]

                    if token_idx + 1 < len(tokens) and tokens[token_idx] == token_l and tokens[token_idx + 1] == token_r:
                        
                        tokens_new.append(token_new) 

                        token_idx += 2

                    else:
                        tokens_new.append(token) 

                        token_idx += 1

                map_word_tokens[word] = tokens_new 

            #print(f"Step {i+1}: merged {pair_max} -> {token_new}")


        self.vocab.sort(key=len)

        self.vocab = self.vocab[::-1]

        return self.vocab
    

    def segment(self, text):
        
        tokenized_text = []

        lines = text.split("\n")
        for line in lines:

            line = line.split(" ")
            
            for word in line:

                word = self.normalize_word(word)
             
                word = word + ' '

                token = self.tokenize_word(word)

                tokenized_text += token
        
        return tokenized_text
    
    def tokenize_word(self, word):

        tokenization = []

        position = 0

        word_len = len(word)

        while position != word_len:
            
            for type in self.vocab:
                type_len = len(type)

                if position + type_len > word_len:
                    continue

                if type == word[position:position + type_len]:
                    tokenization.append(type)
                    position += type_len
                    break
        
        return tokenization
    
    def normalize_word(self, word):
  
        word = word.lower()

        return word
    