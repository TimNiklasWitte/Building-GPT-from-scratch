import torch
from torch.utils.tensorboard import SummaryWriter

import tqdm
import glob

import os
import sys


from NextTokenPredictionDataset import *
from GPT import *


import sys
sys.path.append("./../milestone_1")
from BytePairEncoding import *


NUM_EPOCHS = 32
BATCH_SIZE = 32
NUM_THREADS = 16
BLOCK_SIZE = 8

def main():

    #
    # Device
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    # Hyperparameters controlled via command line
    #
    
    #learning_rate = float(sys.argv[1])
    #k = int(sys.argv[2])
    
    learning_rate = 0.0001
    k = 10

    print(learning_rate, k)
    
    #
    # Logging
    #

    file_path = f"./logs/{learning_rate}_{k}"
    writer = SummaryWriter(file_path)
    
    #
    # Load data
    #

    # Train
    file_path = "./../data/Shakespeare_clean_train.txt"
    with open(file_path) as file:
        corpus_train = file.read()

    # Valid
    file_path = "./../data/Shakespeare_clean_valid.txt"
    with open(file_path) as file:
        corpus_valid = file.read()

    #
    # Tokenization
    #

    bpe = BytePairEncoding(num_types=k)

    bpe.train(corpus_train)

    corpus_train_tokenized = bpe.segment(corpus_train)
    corpus_valid_tokenized = bpe.segment(corpus_valid)

    #
    # Datasets
    #

    train_dataset = NextTokenPredictionDataset(corpus_train_tokenized, block_size=BLOCK_SIZE, vocab=bpe.vocab) 
    valid_dataset = NextTokenPredictionDataset(corpus_valid_tokenized, block_size=BLOCK_SIZE, vocab=bpe.vocab)

    #
    # Data loaders
    #
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True, 
                                               num_workers=NUM_THREADS)
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False, 
                                              num_workers=NUM_THREADS)
    
    #
    # Init GPT
    #

    model = GPT(block_size=BLOCK_SIZE, vocab_size=len(bpe.vocab), learning_rate=learning_rate)
    if torch.cuda.is_available():
        model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total_params: {total_params}")

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

    for epoch in range(NUM_EPOCHS + 1):
        
        print(f"Epoch {epoch}")

        # Epoch 0 = no training steps are performed 
        # test based on train data
        # -> Determinate initial train_loss and train_ppl
        if epoch == 0:
            
            train_loss, train_ppl = model.test(train_loader, device)

        else:

            model.train()

            print("Train")

            for x, targets in tqdm.tqdm(train_loader, position=0, leave=True):
             
                # Transfer data to GPU (if available)
                x, targets = x.to(device), targets.to(device)
             
                # Reset gradients
                model.optimizer.zero_grad()

                # Forward pass
                preds = model(x)

                # Calc loss
                preds = torch.reshape(preds, shape=(-1, model.vocab_size))
                targets = torch.reshape(targets, shape=(-1, ))
                loss = model.cce_loss(preds, targets)

                # Backprob
                loss.backward()

                # Update parameters
                model.optimizer.step()

                #
                # Update metrics
                #

                # Loss
                model.loss_metric.update(loss)

                # Perplexity
                ppl = torch.exp(loss)
                model.ppl_metric.update(ppl)

            train_loss = model.loss_metric.compute()
            train_ppl = model.ppl_metric.compute()

        valid_loss, valid_ppl = model.test(valid_loader, device)

        #
        # Output
        #
        print(f"train_loss: {train_loss}")
        print(f" train_ppl: {train_ppl}")
        print(f"valid_loss: {valid_loss}")
        print(f" valid_ppl: {valid_ppl}")

        #
        # Logging
        #

        writer.add_scalars("Loss",
                            { "Train" : train_loss, "Validation" : valid_loss },
                            epoch)
        
        writer.add_scalars("Perplexity",
                            { "Train" : train_ppl, "Validation" : valid_ppl },
                            epoch)
        
        writer.flush()

        #
        # Save top-3 models
        #

        if len(top_3_ppl_list) == 3:


            for idx, ppl in enumerate(top_3_ppl_list):
                if valid_ppl < ppl:
                    top_3_ppl_list[idx] = valid_ppl

                    # Delete saved model
                    pattern = f"{saved_models_dir}/epoch_*_{ppl}"
                    for filepath in glob.glob(pattern):
                        os.remove(filepath)

    
                    torch.save(model.state_dict(), f"./{saved_models_dir}/epoch_{epoch}_{valid_ppl}")
                    
                    top_3_ppl_list = sorted(top_3_ppl_list, reverse=True)   

                    break
            
        else:

            torch.save(model.state_dict(), f"./{saved_models_dir}/epoch_{epoch}_{valid_ppl}")

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
            return 

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")