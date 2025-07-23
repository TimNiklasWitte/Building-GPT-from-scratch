import matplotlib.pyplot as plt

from BytePairEncoding import *
import numpy as np

def main():

    #
    # Load data
    #

    # shakespeare
    file_path = "./../data/shakespeare.txt"
    with open(file_path) as file:
        corpus = file.read()

    # stop words
    file_path = "./../data/stop_words.txt"
    with open(file_path) as file:
        stop_words = file.read()
     
    stop_word_list = stop_words.split("\n")
    num_stop_words = len(stop_word_list)

    #
    # Measure
    #

    coverage_list = []
    
    for num_types in range(0, 260, 10):

        print(num_types)
        bpe = BytePairEncoding(num_types)

        vocab = bpe.train(corpus)

        cnt_matches = 0
        for type in vocab:
            if type in stop_words:
                cnt_matches += 1
        
        coverage = cnt_matches / num_stop_words

        coverage_list.append(coverage)

    #
    # Plotting
    #

    x_values = list(range(0, 260, 10))

    plt.plot(x_values, coverage_list)
    plt.xlabel("Number of types")
 
    plt.ylabel("Coverage")
    plt.title("Coverage of stop words")

    plt.grid()
    plt.tight_layout()

    plt.savefig("./plots/coverage_stopwords.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")