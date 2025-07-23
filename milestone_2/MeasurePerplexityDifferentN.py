import sys
sys.path.append("./../milestone_1")

import matplotlib.pyplot as plt

from BytePairEncoding import *
from N_Gram_Basic import *

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

    corpus_test_tokenized = bpe.segment(corpus_test)

    perplexity_list = []
    for n in range(1, 7):
        n_gram = N_Gram_Basic(n=n, vocab=bpe.vocab)

        corpus_train_tokenized = bpe.segment(corpus_train)

        n_gram.train(corpus_train_tokenized)

        #
        # Compute perplexity
        #
        
        perplexity = n_gram.perplexity(corpus_test_tokenized)
        perplexity_list.append(perplexity)
        
        print(n, n_gram.perplexity(corpus_test_tokenized))


    x_values = list(range(1, 7))

    plt.plot(x_values, perplexity_list)
    plt.xlabel("n")
 
    plt.ylabel("Perplexity")
    plt.title("Perplexity for different n")

    plt.grid()
    plt.tight_layout()

    plt.savefig("./plots/PerplexityDifferentN.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")