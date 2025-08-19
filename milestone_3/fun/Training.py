import torch
import torchvision
from torchvision.transforms import v2

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data_utils

import tqdm
import datetime
import numpy as np
import os
import sys
from operator import itemgetter 


from NextTokenPredictionDataset import *
from Neural_N_Gram import *


import sys
sys.path.append("./../../milestone_1")
from BytePairEncoding import *

NUM_EPOCHS = 32
BATCH_SIZE = 128
NUM_THREADS = 16

def main():

    #
    # Device
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    # Logging
    #
    
    k = int(sys.argv[1])
    seq_len = int(sys.argv[2])
    
    file_path = f"./logs/{k}/{seq_len}"

    writer = SummaryWriter(file_path)
    
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

    #
    # Tokenization
    #

    bpe = BytePairEncoding(num_types=k)

    bpe.train(corpus_train)

    corpus_train_tokenized = bpe.segment(corpus_train)
    corpus_test_tokenized = bpe.segment(corpus_test)

    #
    # Datasets
    #


    train_dataset = NextTokenPredictionDataset(corpus_train_tokenized, seq_len, bpe.vocab) 
    test_dataset = NextTokenPredictionDataset(corpus_test_tokenized, seq_len, bpe.vocab)

    #
    # Data loaders
    #
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True, 
                                               num_workers=NUM_THREADS)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False, 
                                              num_workers=NUM_THREADS)
    
    #
    # Init Neural N-Gram
    #

    model = Neural_N_Gram(vocab_size=len(bpe.vocab))
    if torch.cuda.is_available():
        model.cuda()
    

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total_params: {total_params}")

    #
    # Train loop
    #
    for epoch in range(NUM_EPOCHS + 1):
        
        print(f"Epoch {epoch}")

        # Epoch 0 = no training steps are performed 
        # test based on train data
        # -> Determinate initial train_loss and train_accuracy
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

        test_loss, test_ppl = model.test(test_loader, device)

        #
        # Output
        #
        print(f"    train_loss: {train_loss}")
        print(f"     train_ppl: {train_ppl}")
        print(f"     test_loss: {test_loss}")
        print(f"      test_ppl: {test_ppl}")

        #
        # Logging
        #

        writer.add_scalars("Loss",
                            { "Train" : train_loss, "Test" : test_loss },
                            epoch)
        
        writer.add_scalars("Perplexity",
                            { "Train" : train_ppl, "Test" : test_ppl },
                            epoch)
        
        writer.flush()

     
    




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")