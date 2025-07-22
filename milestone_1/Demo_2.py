from BytePairEncoding import *
import numpy as np

def main():

    #
    # Load data
    #

    file_path = "./../data/shakespeare.txt"
    with open(file_path) as file:
        corpus_train = file.read()


    file_path = "./../data/bible.txt"
    with open(file_path) as file:
        corpus_test = file.read()

    #
    # Train
    #

    bpe = BytePairEncoding(num_types=50)

    vocab = bpe.train(corpus_train)
    
    print(vocab)
    print("###################")
    print()

    #
    # Tokenize test
    #
    
    corpus_test = corpus_test.split("\n")

    for i, line in enumerate(corpus_test):

        tokens = bpe.segment(line)

        print(line)
        print(tokens)
        print()

        if i == 10:
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")