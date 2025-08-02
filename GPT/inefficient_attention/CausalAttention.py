import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class CausalAttention(torch.nn.Module):

    def __init__(self, block_size, dim, dropout=0.1):
        super(CausalAttention, self).__init__()

        #
        # Query, key, value layer
        #

        self.query_layer = nn.Linear(dim, dim)
        self.key_layer = nn.Linear(dim, dim)
        self.value_layer = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(p=dropout)

        self.sqrt_dim = math.sqrt(dim)

        #
        # Mask generation
        #

        self.mask = torch.ones(size=(block_size, block_size)).cuda()

        for i in range(block_size):
            for j in range(block_size):
                if i < j:
                    self.mask[i][j] = 0

    def forward(self, x):
        
        batch_size, seq_len, dim = x.shape

        x = x.view(batch_size * seq_len, dim)

        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        query = query.view(batch_size, seq_len, dim)
        key = key.view(batch_size, seq_len, dim)
        value = value.view(batch_size, seq_len, dim)

        k_T = key.permute((0,2,1))
        scaled_aff = (query @ k_T) / self.sqrt_dim

        scaled_aff = scaled_aff.view(batch_size, seq_len, seq_len)

        #
        # Causal masking
        #

        scaled_aff = scaled_aff.masked_fill(self.mask == 0, float('-inf'))
        attn = F.softmax(scaled_aff, dim=2)

        attn = self.dropout(attn)


        out = attn @ value

        return out 
        #attn = F.softmax()


