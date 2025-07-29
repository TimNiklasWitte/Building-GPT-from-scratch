import tqdm
from matplotlib import pyplot as plt

import glob
import os
import sys
sys.path.append("./../../milestone_1")
from BytePairEncoding import *

from Dataset import *
from Neural_Bigram import *


NUM_EPOCHS = 250
BATCH_SIZE = 32

def main():

    #
    # Hyperparameters controlled via command line
    #
    
    learning_rate = float(sys.argv[1])
    k = int(sys.argv[2])
   
    print(learning_rate, k)

    #
    # Load data
    #

    # Train
    file_path = "./../../data/Shakespeare_clean_train.txt"
    with open(file_path) as file:
        corpus_train = file.read()

    # Valid
    file_path = "./../../data/Shakespeare_clean_valid.txt"
    with open(file_path) as file:
        corpus_valid = file.read()

    #
    # Tokenization
    #

    bpe = BytePairEncoding(num_types=k)

    bpe.train(corpus_train)

    vocab_size = len(bpe.vocab)

    #
    # Datasets
    #

    corpus_train_tokenized = bpe.segment(corpus_train)
    corpus_valid_tokenized = bpe.segment(corpus_valid)

    train_ds = Dataset(batch_size=BATCH_SIZE, corpus_tokenized=corpus_train_tokenized, vocab=bpe.vocab)
    valid_ds = Dataset(batch_size=BATCH_SIZE, corpus_tokenized=corpus_valid_tokenized, vocab=bpe.vocab)

    #
    # Init Bigram
    #

    bigram = Neural_Bigram(vocab_size=len(bpe.vocab))

    #
    # Store top 3 models
    #

    top_3_ppl_list = []

    saved_models_dir = f"./saved_models/{learning_rate}_{k}"
    os.makedirs(saved_models_dir, exist_ok=True)


    #
    # Early stopping (es)
    #

    cnt_epoch_es = 0
    best_ppl = 10000000

    #
    # Train loop
    #

    train_loss_epoch_list = []
    train_ppl_epoch_list = []

    valid_loss_epoch_list = []
    valid_ppl_epoch_list = []

    for epoch in range(NUM_EPOCHS + 1):

        print(f"Epoch {epoch}")

        # Epoch 0 = no training steps are performed 
        # test based on train data
        # -> Determinate initial train_loss and train_ppl
        
        if epoch == 0:
            train_loss, train_ppl = bigram.test(train_ds)

        else:
            
            # Shuffle dataset
            train_ds.shuffle()

            print("Train")

            train_loss_list = []
            train_ppl_list = []
            for batch in tqdm.tqdm(train_ds.corpus_tokenID, position=0, leave=True):

                # Get input and target
                x = batch[:, 0]
                targets = batch[:, 1]

                # Forward pass
                y_hat = bigram.forward(x)

                # Compute gradient
                y = np.zeros(shape=(BATCH_SIZE, vocab_size))
                y[np.arange(len(targets)), targets] = 1

                dL_dEmbedding = y_hat - y

                # Update embeddings
                for idx, x_token_idx in enumerate(x):
                    bigram.embedding[x_token_idx] = bigram.embedding[x_token_idx] - learning_rate * dL_dEmbedding[idx]

                #
                # Update metrics
                #

                # Loss
                loss = bigram.cce_loss(y_hat, targets)
                train_loss_list.append(loss)

                # Perplexity
                ppl = np.exp(loss)
                train_ppl_list.append(ppl)


            train_loss = np.average(train_loss_list)
            train_ppl = np.average(train_ppl_list)
        
        valid_loss, valid_ppl = bigram.test(valid_ds)

        #
        # Output
        #
        print(f"train_loss: {train_loss}")
        print(f" train_ppl: {train_ppl}")
        print(f"valid_loss: {valid_loss}")
        print(f" valid_ppl: {valid_ppl}")


        train_loss_epoch_list.append(train_loss)
        train_ppl_epoch_list.append(train_ppl)

        valid_loss_epoch_list.append(valid_loss)
        valid_ppl_epoch_list.append(valid_ppl)


        #
        # Save top-3 models
        #

        if len(top_3_ppl_list) == 3:


            for idx, ppl in enumerate(top_3_ppl_list):
                if valid_ppl < ppl:

                    top_3_ppl_list[idx] = valid_ppl

                    # Delete saved model
                    pattern = f"{saved_models_dir}/epoch_*_{ppl}.npy"
                    for filepath in glob.glob(pattern):
                        
                        os.remove(filepath)

    
                    bigram.save(f"./{saved_models_dir}/epoch_{epoch}_{valid_ppl}")
                    
                    top_3_ppl_list = sorted(top_3_ppl_list, reverse=True)   

                    break
            
        else:

            bigram.save(f"./{saved_models_dir}/epoch_{epoch}_{valid_ppl}")

            top_3_ppl_list.append(valid_ppl)
            top_3_ppl_list = sorted(top_3_ppl_list, reverse=True) 

    

        #
        # Early stopping
        #

        if valid_ppl < best_ppl:
            cnt_epoch_es =  0
            best_ppl = valid_ppl
        else:
            cnt_epoch_es += 1

        # Tolerance threshold (patience) reached -> stop training
        if cnt_epoch_es == 3:
            break 
    
    #
    # Plotting
    #

    fig, axs = plt.subplots(nrows=1, ncols=2)
    
    axs[0].plot(train_loss_epoch_list, label="train")
    axs[0].plot(valid_loss_epoch_list, label="valid")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title("Loss")

    axs[1].plot(train_ppl_epoch_list, label="train")
    axs[1].plot(valid_ppl_epoch_list, label="valid")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Perplexity")
    axs[1].grid()
    axs[1].legend()
    axs[1].set_title("Perplexity")
    
    plt.tight_layout()

    plt.savefig(f"./plots/{learning_rate}_{k}.png", dpi=200)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")