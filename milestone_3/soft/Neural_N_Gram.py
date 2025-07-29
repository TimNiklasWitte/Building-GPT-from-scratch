import torch
import torch.nn as nn

import tqdm

from torchmetrics import MeanMetric

class Neural_N_Gram(torch.nn.Module):

    def __init__(self, n, vocab_size, learning_rate):
        super(Neural_N_Gram, self).__init__()

        self.n = n

        self.embedding_dim = 32

        self.embedd = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)

        #
        # Processing more than 1 word
        # 
        if n > 2:
            self.hidden_size = 64
            self.lstm = nn.LSTM(
                            input_size=self.embedding_dim, 
                            hidden_size=self.hidden_size, 
                            num_layers=1,
                            batch_first=True
                        )
        else:
            
            # More complex embeddings
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, vocab_size)
            )
        

        self.cce_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.ppl_metric = MeanMetric()
        self.loss_metric = MeanMetric()


    def forward(self, x):
 
        # x: (bs, seq_len, vocab_size)

        x = self.embedd(x)

        # x: (bs, seq_len, embedding_dim)

        #
        # Processing more than 1 word
        #

        if 2 < self.n:

            x, (h_n, c_n) = self.lstm(x)

            # x: (bs, seq_len, hidden_size)
        
            # Consider only last output
            x = x[:, -1, :] 

            # x: (bs, hidden_size)

            x = self.classifier(x)

        else:
            
            # Consider only last output
            x = x[:, -1, :] 

            # x: (bs, embedding_dim)

            x = self.classifier(x)

        return x

    
    @torch.no_grad
    def test(self, test_loader, device):

        self.eval()

        self.loss_metric.reset()
        self.ppl_metric.reset()

        print("Test")

        for x, targets in tqdm.tqdm(test_loader, position=0, leave=True):
  
            x, targets = x.to(device), targets.to(device)

            preds = self(x)

            # Loss
            loss = self.cce_loss(preds, targets)

            self.loss_metric.update(loss)

            # Perplexity
            ppl = torch.exp(loss)
            self.ppl_metric.update(ppl)

        test_loss = self.loss_metric.compute()
        test_ppl = self.ppl_metric.compute()

        self.loss_metric.reset()
        self.ppl_metric.reset()
        
        return test_loss, test_ppl 
        