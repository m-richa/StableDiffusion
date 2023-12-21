import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

class selfAttention(nn.Module):
    
    def __init__(self, n_heads, d_embed, in_bias=False, out_bias=False):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_bias)
        self.out_proj = nn.linear(d_embed, d_embed, bias = out_bias)
        
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads
        
    def forward(self, x, mask=False):
        
        batch_size, seq_len, d_embed = x.shape
        temp_shape = (batch_size, seq_len, self.n_heads, self.d_heads)
        
        #b,s,d -> b,s,3*d -> b,s,d
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        #b,s,d -> b,s,h,d/h -> b,h,s,d/h
        q = q.view(*temp_shape).transpose(1,2)
        k = k.view(*temp_shape).transpose(1,2)
        v = v.view(*temp_shape).transpose(1,2)
        
        #b,h,s,d/h -> b,h,s,s
        weights = q @ k.transpose(-1,-2)
        
        if mask:
            mask = torch.ones_like(weights, dtype='bool').triu_(1)
            weights.masked_fill_(mask, float('-inf'))
            
        weight /= math.sqrt(self.d_heads)
            
        