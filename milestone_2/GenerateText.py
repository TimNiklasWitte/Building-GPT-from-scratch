import sys
sys.path.append("./../milestone_1")

import os

from BytePairEncoding import *
from N_Gram_Basic import *
from N_Gram_Advanced import *


def generate(context, bpe, model, get_distri, sample=False):
    
    context_tokenized = bpe.segment(context)

    token_ids = [model.map_token_to_id[id] for id in context_tokenized]

    # n = 1 -> unigram P(wt​​)           -> window_size = 0 no window
    # n = 2 -> bigram  P(wt​∣​wt−1​)      -> window_size = 1
    # n = 3 -> trigram P(wt​∣wt−2​,wt−1​) -> window_size = 2

    window_size = model.n - 1

    stop_token_id = model.map_token_to_id["."]
    new_line_token_id = model.map_token_to_id['\n']

    cnt_tokens = 0

    while cnt_tokens != 100:
        
        if window_size == 0:
            token_ids_window = []
        else:
            token_ids_window = token_ids[-window_size:]

        distri = get_distri(token_ids_window)
        
        #
        # Mask out new lines
        #
        distri[new_line_token_id] = 0 
        
        # Normalize
        distri = distri / np.sum(distri)

        if sample:
            next_token_id = np.random.choice(model.vocab_size, p=distri)
        else:
            next_token_id = np.argmax(distri)

        token_ids.append(next_token_id)

        cnt_tokens += 1
        
        if next_token_id == stop_token_id:
            break

    
    
    # Token ids -> tokens
    
    tokens = [model.vocab[token_id] for token_id in token_ids]

    # tokens -> text
    text = "".join(tokens)

    return text

def main():
    
    #
    # Load data
    #

    # Train
    file_path = "./../data/Shakespeare_clean_train.txt"
    with open(file_path) as file:
        corpus_train = file.read()

    #
    # Train BytePairEncoding
    #

    bpe = BytePairEncoding(num_types=100)

    bpe.train(corpus_train)

    #
    # Train N_Gram
    #

    corpus_train_tokenized = bpe.segment(corpus_train)

    context = "hello julia"

    for n in range(1, 7):

        root = f"./generated_texts/{n}"
        os.makedirs(root, exist_ok=True)

        #
        # Laplace smoothing
        #

        n_gram = N_Gram_Basic(n=n, vocab=bpe.vocab)
        n_gram.train(corpus_train_tokenized)

        # argmax
        text = generate(context, 
                        bpe, 
                        n_gram, 
                        n_gram.get_distri, 
                        sample=False
                    )

        path = f"{root}/laplace_argmax.txt"
        with open(path, "w") as file:
            file.write(text)
        
        # sample
        text = generate(context, 
                        bpe, 
                        n_gram, 
                        n_gram.get_distri, 
                        sample=True
                    )
        
        path = f"{root}/laplace_sample.txt"
        with open(path, "w") as file:
            file.write(text)


        #
        # Interpolation
        #

        n_gram = N_Gram_Advanced(n=n, vocab=bpe.vocab)
        n_gram.train(corpus_train_tokenized)

        weights = np.load(f"./weights/{n}.npy")
        n_gram.weights = weights 

        # argmax
        text = generate(context, 
                        bpe, 
                        n_gram, 
                        lambda x: n_gram.get_distri(n_gram.get_prob_interpolation, x), 
                        sample=False
                    )
        
        path = f"{root}/interpolation_argmax.txt"
        with open(path, "w") as file:
            file.write(text)
        
        # sample
        text = generate(context, 
                        bpe, 
                        n_gram, 
                        lambda x: n_gram.get_distri(n_gram.get_prob_interpolation, x), 
                        sample=True
                    )

        path = f"{root}/interpolation_sample.txt"
        with open(path, "w") as file:
            file.write(text)

        #
        # Backoff
        #

        # reuse n_gram from Interpolation

        # argmax
        text = generate(context, 
                        bpe, 
                        n_gram, 
                        lambda x: n_gram.get_distri(n_gram.get_prob_backoff_logic, x), 
                        sample=False
                    )
        
        path = f"{root}/backoff_logic_argmax.txt"
        with open(path, "w") as file:
            file.write(text)

        # sample
        text = generate(context, 
                        bpe, 
                        n_gram, 
                        lambda x: n_gram.get_distri(n_gram.get_prob_backoff_logic, x), 
                        sample=True
                    )

        path = f"{root}/backoff_logic_sample.txt"
        with open(path, "w") as file:
            file.write(text)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")