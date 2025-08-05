import matplotlib.pyplot as plt

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
    # Unique words
    #

    lines = corpus.split("\n")

    word_set = set()
    for line in lines:

            line = line.split(" ")
            
            for word in line:

                word = word.lower()
    
                word_set.add(word)

    #
    # Measure
    #

    branch_factor_list = []

    for num_types in range(0, 260, 10):

        print(num_types)
        bpe = BytePairEncoding(num_types)

        bpe.train(corpus)

        branch_factor_list_tmp = []
        for word in word_set:
            tokens = bpe.tokenize_word(word)
            num_tokens = len(tokens)
            branch_factor_list_tmp.append(num_tokens)
        
        branch_factor = np.average(branch_factor_list_tmp)

        branch_factor_list.append(branch_factor)

    x_values = list(range(0, 260, 10))

    plt.plot(x_values, branch_factor_list)
    plt.xlabel("Number of types")
 
    plt.ylabel("Average branch factor")
    plt.title("Average branch factor of vocabulary")

    plt.grid()
    plt.tight_layout()

    plt.savefig("./plots/branch_factor.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")