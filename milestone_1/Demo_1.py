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

    corpus = np.array(corpus.split("\n")) # create array of lines
    num_lines = len(corpus) # get number of lines
  
    
    idxs = np.random.permutation(num_lines) # shuffle indices
    np.random.seed(42) # set seed for reproducibility

    train_size = int(num_lines*0.99)

    idxs_train = idxs[:train_size] # select first 99% of indices
    

   
    corpus_train = corpus[idxs_train] # select first 99% of lines

    idxs_test = idxs[train_size:] # select last 1% of indices
    corpus_test = corpus[idxs_test] # select last 1% of lines


    print("Train corpus size:", len(corpus_train))
    print(" Test corpus size:", len(corpus_test))

    # Convert back to text

    corpus_train = list(corpus_train) # convert to list of strings
    corpus_train = "\n".join(corpus_train) # convert to single string
    
    corpus_test = list(corpus_test) # convert to list of strings
    
    #
    # Train
    #

    bpe = BytePairEncoding(num_types=50) # number of types to learn

    vocab = bpe.train(corpus_train) # train BPE model
    
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