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
    n_gram.weights = np.array([0, 1, 0, 0])

    corpus_train_tokenized = bpe.segment(corpus_train)

    n_gram.train(corpus_train_tokenized)
    
    #
    # Compute perplexity
    #

    corpus_test_tokenized = bpe.segment(corpus_test)

    perplexity = n_gram.perplexity(corpus_test_tokenized, n_gram.get_prob_backoff_logic)

    print("perplexity: ", perplexity)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")