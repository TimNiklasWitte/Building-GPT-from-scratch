import torch
import torch.nn as nn

from MultiHeadAttention import *

class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, n_heads, block_size, dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()

        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            MultiHeadAttention(n_heads=n_heads, block_size=block_size, dim=dim, dropout=dropout),
            nn.Dropout(dropout)
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        
        x = x + self.attn(x)

        x = x + self.ffn(x)

        return x
