import sys
sys.path.append("./../../milestone_1")

import os 

from BytePairEncoding import *
from Dataset import *
from Neural_Bigram import *

BATCH_SIZE = 32

def main():

    #
    # Load data
    #

    # Train
    file_path = "./../../data/Shakespeare_clean_train.txt"
    with open(file_path) as file:
        corpus_train = file.read()

    # Test
    file_path = "./../../data/Shakespeare_clean_test.txt"
    with open(file_path) as file:
        corpus_test = file.read()

    learning_rate_list = [0.0001, 0.0005, 0.001]
    k_list = [100, 200, 300, 500]


    best_test_ppl = 1000000 
    best_test_ppl_config = None

    for k in k_list:

        #
        # Tokenization
        #

        bpe = BytePairEncoding(num_types=k)

        bpe.train(corpus_train)

        #
        # Init Bigram
        #

        bigram = Neural_Bigram(vocab_size=len(bpe.vocab))

        #
        # Datasets
        #

        corpus_test_tokenized = bpe.segment(corpus_test)

        test_ds = Dataset(batch_size=BATCH_SIZE, corpus_tokenized=corpus_test_tokenized, vocab=bpe.vocab)
        
      
        for learning_rate in learning_rate_list:
            
            root = f"./saved_models/{learning_rate}_{k}/"
            files = os.listdir(root)
            for file in files:
                path = root + file 

                epoch = int(file.split("_")[1])

                if epoch == 250:

                    bigram.load(path)

                    test_loss, test_ppl = bigram.test(test_ds)
                    
                    print(f"Learning rate: {learning_rate}, k: {k} -> test perplexity: {test_ppl}")

                    if test_ppl < best_test_ppl:
                        best_test_ppl = test_ppl
                        
                        best_test_ppl_config = (learning_rate, k)

                    

    print("###################")
    print("Best:")

    learning_rate, k = best_test_ppl_config

    print(f"Learning rate: {learning_rate}, k: {k} -> test perplexity: {best_test_ppl}")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")