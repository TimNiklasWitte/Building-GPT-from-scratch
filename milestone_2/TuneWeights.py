import sys
sys.path.append("./../milestone_1")

from N_Gram_Advanced import *
from BytePairEncoding import *

def main():

    #
    # Load data
    #

    # Train
    file_path = "./../data/Shakespeare_clean_train.txt"
    with open(file_path) as file:
        corpus_train = file.read()

    # Val
    file_path = "./../data/Shakespeare_clean_valid.txt"
    with open(file_path) as file:
        corpus_val = file.read()

    bpe = BytePairEncoding(num_types=100)

    vocab = bpe.train(corpus_train)

    #
    # Train N-Gram
    #
    
    corpus_val_tokenized = bpe.segment(corpus_val)

    for n in range(1, 7):
        print(f"n = {n}")

        n_gram = N_Gram_Advanced(n=n, vocab=vocab)
    
        corpus_train_tokenized = bpe.segment(corpus_train)

        n_gram.train(corpus_train_tokenized)
        

        #
        # Tune weights
        #

        # order: 4-gram, 3-gram, 2-gram, 1-gram
        num_candidates = 10
        weights_candidates = np.random.rand(num_candidates, n)
        # Normalize
        weights_candidates = weights_candidates / np.sum(weights_candidates, axis=1, keepdims=True)

    
        perplexity_list = []

        for weights_candidate in weights_candidates:
        
            n_gram.weights = weights_candidate

            perplexity = n_gram.perplexity(corpus_val_tokenized, n_gram.get_prob_interpolation)

            perplexity_list.append(perplexity)

            print("candidate", weights_candidate, "perplexity:", perplexity)
        best_candidate_idx = np.argmin(perplexity_list)
        best_weights = weights_candidates[best_candidate_idx]

        np.save(f"./weights/{n}", best_weights)
        print("###################")
        print("best: ", best_weights, "perplexity:", np.min(perplexity_list))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")