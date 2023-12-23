import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention, CrossAttention
from clip import CLIP


class TimeEmbedding(nn.Module):
    
    def __init__(self, n_embed):
        super().__init__()
        
        self.linear1 = nn.Linear(n_embed, 4*n_embed)
        self.linear2 = nn.Linear(4*n_embed, 4*n_embed)
        
    def forward(self, x):
        
        # x: 1, n_embed
        
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        
        return x
    

class UNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Module([
            SwitchSequential()
        ])


class Diffusion(nn.Module):
    
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNetFinal(320, 4)
        
    def forward(self, latent, context, time):
        
        #latent: b,4,h/8,w/8
        #context: b, seq_len, d_embed=768
        #time: 1,320
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # b,4,h/8,w/8 -> b,320,h/8,w/8
        output = self.unet(latent, context, time)
        
        # b,320,h/8,w/8 -> b,4,h/8,w/8
        output = self.final(output)
        
        return output