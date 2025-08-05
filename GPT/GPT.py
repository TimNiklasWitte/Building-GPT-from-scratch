import torch
import torch.nn as nn

import tqdm

from torchmetrics import MeanMetric

from TransformerDecoderBlock import *


class GPT(torch.nn.Module):

    def __init__(self, block_size, vocab_size, emb_dim):
        super(GPT, self).__init__()

        self.vocab_size = vocab_size

        #
        # Token embedding
        #

        self.emb_dim = emb_dim
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.emb_dim)

        #
        # Positional embedding
        #

        self.pos_embedding = nn.Parameter(
            torch.randn(size=(1, block_size, self.emb_dim))
        )

        #
        # Dropout
        #

        self.dropout = nn.Dropout(p=0.1)

        #
        # Layers
        #

        self.decoder_blocks = nn.Sequential(
            TransformerDecoderBlock(n_heads=4, block_size=block_size, dim=self.emb_dim, dropout=0.1),
            TransformerDecoderBlock(n_heads=4, block_size=block_size, dim=self.emb_dim, dropout=0.1),
            TransformerDecoderBlock(n_heads=4, block_size=block_size, dim=self.emb_dim, dropout=0.1)
        )

        self.layer_norm = nn.LayerNorm(self.emb_dim)
        self.lm_head = nn.Linear(self.emb_dim, vocab_size)

        #
        # Optimization
        #

        self.cce_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        #
        # Metrics
        #

        self.ppl_metric = MeanMetric()
        self.loss_metric = MeanMetric()


    def forward(self, x):
        
        batch_size, seq_len = x.shape
        
        x = self.token_embedding(x) + self.pos_embedding

        x = self.dropout(x)

        x = self.decoder_blocks(x)

        x = self.layer_norm(x)

        x = self.lm_head(x)

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

            preds = torch.reshape(preds, shape=(-1, self.vocab_size))
            targets = torch.reshape(targets, shape=(-1, ))
            
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
        