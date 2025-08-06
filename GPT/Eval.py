import sys
sys.path.append("./../milestone_1")

import os 

from BytePairEncoding import *
from NextTokenPredictionDataset import *
from GPT import *

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
    file_path = "./../data/Shakespeare_clean_train.txt"
    with open(file_path) as file:
        corpus_train = file.read()

    # Test
    file_path = "./../data/Shakespeare_clean_test.txt"
    with open(file_path) as file:
        corpus_test = file.read()

    embedd_dim_list = [16, 32, 64]
    k_list = [50, 150, 250, 1000]


    best_test_ppl = 1000000 
    best_test_ppl_config = None

    for k in k_list:

        #
        # Tokenization
        #

        bpe = BytePairEncoding(num_types=k)

        bpe.train(corpus_train)

        #
        # Dataset
        #

        corpus_test_tokenized = bpe.segment(corpus_test)

        test_dataset = NextTokenPredictionDataset(corpus_test_tokenized, block_size=64, vocab=bpe.vocab)
        
        #
        # Data loader
        #
    
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=BATCH_SIZE, 
                                                  shuffle=False, 
                                                  num_workers=NUM_THREADS)
        
        for emb_dim in embedd_dim_list:
            
            #
            # Init GPT
            #

            # learning rate does not matter for evaluation!
            model = GPT(block_size=64, vocab_size=len(bpe.vocab), emb_dim=emb_dim)
         

            root = f"./saved_models/{emb_dim}_{k}/"
            file = os.listdir(root)[-1]
        
            path = root + file 
            
            model.load_state_dict(torch.load(path, weights_only=True))

            if torch.cuda.is_available():
                model.cuda()

            model.eval()

            test_loss, test_ppl = model.test(test_loader, device)
                    
            print(f"Embedding dim: {emb_dim}, k: {k} -> test perplexity: {test_ppl}")

            if test_ppl < best_test_ppl:
                best_test_ppl = test_ppl
                        
                best_test_ppl_config = (emb_dim, k)

                    

    print("###################")
    print("Best:")

    learning_rate, k = best_test_ppl_config

    print(f"Learning rate: {learning_rate}, k: {k} -> test perplexity: {best_test_ppl}")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")