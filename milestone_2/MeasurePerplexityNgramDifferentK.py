import sys
sys.path.append("./../milestone_1")

import matplotlib.pyplot as plt

from BytePairEncoding import *
from N_Gram_Advanced import *

N = 6

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


    perplexity_list = []

    for k in range(0, 250, 10):
    
        #
        # Train BytePairEncoding
        #

        bpe = BytePairEncoding(num_types=k)

        bpe.train(corpus_train)

        #
        # Train N gram
        #

        n_gram = N_Gram_Advanced(n=6, vocab=bpe.vocab)

        corpus_train_tokenized = bpe.segment(corpus_train)

        n_gram.train(corpus_train_tokenized)

        #
        # Compute perplexity
        #
        
        corpus_test_tokenized = bpe.segment(corpus_test)

        perplexity = n_gram.perplexity(corpus_test_tokenized, n_gram.get_prob_backoff_logic)
        perplexity_list.append(perplexity)
        
        print(k, perplexity)


    x_values = list(range(0, 250, 10))

    plt.plot(x_values, perplexity_list)
    plt.xlabel("k")
 
    plt.ylabel("Perplexity")
    plt.title(f"Perplexity of {N}-Gram (backoff) for different k")

    plt.grid()
    plt.tight_layout()

    plt.savefig(f"./plots/Perplexity{N}_GramDifferentK.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")