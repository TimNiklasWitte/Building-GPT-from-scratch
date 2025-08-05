import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class CausalMultiHeadAttention(torch.nn.Module):

    def __init__(self, n_heads, block_size, dim, dropout):
        super(CausalMultiHeadAttention, self).__init__()

        assert dim % n_heads == 0

        self.query_key_value_layer = nn.Linear(dim, 3*dim)

        self.dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(dim, dim)

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.sqrt_head_dim = math.sqrt(self.head_dim)

        #
        # Mask generation
        #

        mask = torch.tril(
            torch.ones(size=(block_size, block_size))
        )

        # add dummy dim for batch dim and sequence dim
        mask = mask.view(1, 1, block_size, block_size)

        self.register_buffer("mask", mask)

    
    def forward(self, x):

        batch_size, seq_len, dim = x.shape

        query_key_value = self.query_key_value_layer(x)

        # query_key_value: (batch_size, seq_len, 3*dim)

        query, key, value  = query_key_value.split(dim, dim=2)

        #
        # Multi-head
        #

        # query, key, value: (batch_size, seq_len, dim)

        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # query, key, value: (batch_size, seq_len, n_heads, head_dim)

        query = query.permute(0, 2, 1, 3)
        key_T = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        # query, value: (batch_size, n_heads, seq_len, head_dim)
        # key_T: (batch_size, n_heads, head_dim, seq_len)

        #
        # Affinites
        #

        aff = query @ key_T

        # aff: (batch_size, n_heads, seq_len, seq_len)

        # Scaling
        aff = aff / self.sqrt_head_dim

        # aff: (batch_size, n_heads, seq_len, seq_len)

        #
        # Masking
        #

        aff = aff.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))

        # aff: (batch_size, n_heads, seq_len, seq_len)

        #
        # Softmax 
        #

        attn = F.softmax(aff,dim=3)

        # attn: (batch_size, n_heads, seq_len, seq_len)

        #
        # Dropout
        #

        attn = self.dropout(attn)
        
        # attn: (batch_size, n_heads, seq_len, seq_len)

        out = attn @ value

        # out: (batch_size, n_heads, seq_len, head_dim)

        #
        # Merge heads
        #

        out = out.permute(0, 2, 1, 3)

        # out: (batch_size, seq_len, n_heads, head_dim)

        out = torch.reshape(out, shape=(batch_size, seq_len, dim))

        # out: (batch_size, seq_len, n_heads * head_dim)
        #    = (batch_size, seq_len, dim)

        out = self.linear(out)
    
        return out