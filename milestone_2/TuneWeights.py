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

    # Test
    file_path = "./../data/Shakespeare_clean_test.txt"
    with open(file_path) as file:
        corpus_test = file.read()

    bpe = BytePairEncoding(num_types=100)

    vocab = bpe.train(corpus_train)

    #
    # Train N-Gram
    #
    
    n_gram = N_Gram_Advanced(n=4, vocab=vocab)

    corpus_train_tokenized = bpe.segment(corpus_train)

    n_gram.train(corpus_train_tokenized)
    

    #
    # Tune weights
    #

    # order: 4-gram, 3-gram, 2-gram, 1-gram
    weights_candidates = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.3, 0.3, 0.3],
        [0, 0, 0.5, 0.5],
        [0, 0, 0.75, 0.25],
        [0, 0, 0.25, 0.75],
    ])

    corpus_test_tokenized = bpe.segment(corpus_test)


    perplexity_list = []

    for weights_candidate in weights_candidates:
        
        n_gram.weights = weights_candidate

        perplexity = n_gram.perplexity(corpus_test_tokenized, n_gram.get_prob_interpolation)

        perplexity_list.append(perplexity)
        print(perplexity, weights_candidate)

    best_candidate_idx = np.argmin(perplexity_list)
    
    print()
    print("Winner: ", weights_candidates[best_candidate_idx], "perplexity:", np.min(perplexity_list))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")