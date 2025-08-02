import torch
import torch.nn as nn
import torch.nn.functional as F

from CausalAttention import *

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, n_heads, block_size, dim, dropout):
        super(MultiHeadAttention, self).__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads

        self.dim = dim

        self.head_dim = dim // self.n_heads
        self.attention_list = nn.ModuleList(
            [CausalAttention(block_size=block_size, dim=self.head_dim, dropout=dropout) for _ in range(n_heads)]
        )
      
        self.layer = nn.Linear(dim, dim)


    def forward(self, x):
        
        batch_size, seq_len, dim = x.shape

        
        out_list = [
            attention(x[:, :, idx*self.head_dim:(idx+1)*self.head_dim]) for idx, attention in enumerate(self.attention_list)
        ]

        out = torch.stack(out_list, dim=-1)

        out = torch.reshape(out, shape=(batch_size * seq_len, dim))

        out = self.layer(out)

        out = torch.reshape(out, shape=(batch_size, seq_len, self.dim))

        return out