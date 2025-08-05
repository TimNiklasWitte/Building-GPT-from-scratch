import sys
sys.path.append("./../milestone_1")

import matplotlib.pyplot as plt

from BytePairEncoding import *
from N_Gram_Basic import *
from N_Gram_Advanced import *

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

    #
    # Train BytePairEncoding
    #

    bpe = BytePairEncoding(num_types=100)

    bpe.train(corpus_train)

    #
    # Train N_Gram
    #

    corpus_train_tokenized = bpe.segment(corpus_train)
    corpus_test_tokenized = bpe.segment(corpus_test)

    perplexity_laplace_list = []
    perplexity_interpolation_list = []
    perplexity_backoff_list = []
    for n in range(1, 7):

        #
        # Laplace smoothing
        #
        n_gram = N_Gram_Basic(n=n, vocab=bpe.vocab)
        n_gram.train(corpus_train_tokenized)

        # Compute perplexity
        perplexity = n_gram.perplexity(corpus_test_tokenized)
        perplexity_laplace_list.append(perplexity)
        
        #
        # Interpolation
        #

        n_gram = N_Gram_Advanced(n=n, vocab=bpe.vocab)
        n_gram.train(corpus_train_tokenized)

        weights = np.load(f"./weights/{n}.npy")
        n_gram.weights = weights 

        # Compute perplexity
        perplexity = n_gram.perplexity(corpus_test_tokenized, n_gram.get_prob_interpolation)
        perplexity_interpolation_list.append(perplexity)

        #
        # Backoff
        #

        # reuse n_gram from Interpolation

        # Compute perplexity
        perplexity = n_gram.perplexity(corpus_test_tokenized, n_gram.get_prob_backoff_logic)
        perplexity_backoff_list.append(perplexity)

    x_values = list(range(1, 7))

    plt.plot(x_values, perplexity_laplace_list, label="Laplace smoothing")
    plt.plot(x_values, perplexity_interpolation_list, label="Interpolation")
    plt.plot(x_values, perplexity_backoff_list, label="Backoff")
    
    plt.xlabel("n")
 
    plt.ylabel("Perplexity")
    plt.title("Perplexity for different n")

    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig("./plots/PerplexityDifferentN.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")