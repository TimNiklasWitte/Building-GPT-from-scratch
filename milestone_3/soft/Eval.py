import sys
sys.path.append("./../../milestone_1")

import os 

from BytePairEncoding import *
from NextTokenPredictionDataset import *
from Neural_N_Gram import *

BATCH_SIZE = 32
NUM_THREADS = 16

def main():

    #
    # Device
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # learning rate does not matter for evaluation!
        bigram = Neural_N_Gram(n=2, vocab_size=len(bpe.vocab), learning_rate=0.001) 
        if torch.cuda.is_available():
            bigram.cuda()

        #
        # Dataset
        #

        corpus_test_tokenized = bpe.segment(corpus_test)

        test_dataset = NextTokenPredictionDataset(corpus_test_tokenized, seq_len=1, vocab=bpe.vocab)
        
        #
        # Data loader
        #
    
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=BATCH_SIZE, 
                                                  shuffle=False, 
                                                  num_workers=NUM_THREADS)
        
        for learning_rate in learning_rate_list:
            
            root = f"./saved_models/{learning_rate}_{k}/"
            file = os.listdir(root)[-1]
        
            path = root + file 

            bigram.load_state_dict(torch.load(path, weights_only=True))

            test_loss, test_ppl = bigram.test(test_loader, device)
                    
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