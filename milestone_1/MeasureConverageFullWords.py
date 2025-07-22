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

    num_words = len(word_set)

    coverage_list = []

    for num_types in range(0, 260, 10):

        print(num_types)
        bpe = BytePairEncoding(num_types)

        vocab = bpe.train(corpus)

        cnt_matches = 0
        for type in vocab:
            if type in word_set:
                cnt_matches += 1
        
        if num_types == 0:
            coverage = 0
        else:
            coverage = cnt_matches / num_words

        coverage_list.append(coverage)

    x_values = list(range(0, 260, 10))

    plt.plot(x_values, coverage_list)
    plt.xlabel("Number of types")
 
    plt.ylabel("Coverage [%]")
    plt.title("Coverage of full words")

    plt.grid()
    plt.tight_layout()

    plt.savefig("./plots/coverage_fullwords.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")