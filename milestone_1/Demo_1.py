from BytePairEncoding import *
import numpy as np

def main():

    #
    # Load data
    #

    file_path = "./../data/shakespeare.txt"
    with open(file_path) as file:
        corpus = file.read()

    #
    # Split into train and test
    #

    corpus = np.array(corpus.split("\n"))
    num_lines = len(corpus)
  
    
    idxs = np.random.permutation(num_lines)

    train_size = int(num_lines*0.99)

    idxs_train = idxs[:train_size]

   
    corpus_train = corpus[idxs_train]

    idxs_test = idxs[train_size:]
    corpus_test = corpus[idxs_test]


    print("Train corpus size:", len(corpus_train))
    print(" Test corpus size:", len(corpus_test))

    # Convert back to text

    corpus_train = list(corpus_train)
    corpus_train = "\n".join(corpus_train)
    
    corpus_test = list(corpus_test)
    
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