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


    perplexity_list = []

    for k in range(0, 250, 5):

        #
        # Train BytePairEncoding
        #

        bpe = BytePairEncoding(num_types=k)

        bpe.train(corpus_train)

        #
        # Train Bigram
        #

        bigram = N_Gram_Basic(n=2, vocab=bpe.vocab)

        corpus_train_tokenized = bpe.segment(corpus_train)

        bigram.train(corpus_train_tokenized)

        #
        # Compute perplexity
        #
        
        corpus_test_tokenized = bpe.segment(corpus_test)

        perplexity = bigram.perplexity(corpus_test_tokenized)
        perplexity_list.append(perplexity)
        
        print(k, bigram.perplexity(corpus_test_tokenized))


    x_values = list(range(0, 250, 5))

    plt.plot(x_values, perplexity_list)
    plt.xlabel("k")
 
    plt.ylabel("Perplexity")
    plt.title("Perplexity of Bigram for different k")

    plt.grid()
    plt.tight_layout()

    plt.savefig("./plots/PerplexityBigramDifferentK.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")