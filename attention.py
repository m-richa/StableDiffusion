import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads, d_embed, in_bias=True, out_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_bias)
        self.out_proj = nn.linear(d_embed, d_embed, bias = out_bias)
        
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads
        
    def forward(self, x, mask=False):
        
        input_shape = x.shape
        batch_size, seq_len, d_embed = x.shape
        temp_shape = (batch_size, seq_len, self.n_heads, self.d_heads)
        
        #b,s,d -> b,s,3*d -> b,s,d
        # qkv = self.in_proj(x)
        # q = qkv[:,:,0:d_embed]
        # k = qkv[:,:,d_embed:2*d_embed]
        # v = qkv[:,:,2*d_embed:3*d_embed]
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        #b,s,d -> b,s,h,d/h -> b,h,s,d/h
        q = q.view(temp_shape).transpose(1,2)
        k = k.view(temp_shape).transpose(1,2)
        v = v.view(temp_shape).transpose(1,2)
        
        #b,h,s,d/h -> b,h,s,s
        weights = q @ k.transpose(-1,-2)
        
        if mask:
            mask = torch.ones_like(weights, dtype='bool').triu(1)
            weights.masked_fill_(mask, float('-inf'))
            
        weight /= math.sqrt(self.d_heads)
        weights = F.softmax(weights, dim=-1)
        
        #b,h,s,s @ b,h,s,d/h --> b,h,s,d/h
        out = weights @ v
        
        out = out.transpose(1,2).contiguous()
        out = out.reshape(input_shape)
        
        out = self.out_proj(out)
        
        return out
            
        