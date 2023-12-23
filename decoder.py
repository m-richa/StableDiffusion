import torch
import torch.nn as nn
from torch.nn import Functional as F

from attention import SelfAttention

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x):
        
        #x: B, C, H, W
        
        init_x = x
        
        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        x = x + self.skip(init_x)
        
        return x
        
class AttentionBlock(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x):
        
        #x: B, C, H, W
        
        residue = x
        
        x = self.groupnorm(x)
        
        n, c, h, w = x.shape
        
        # Attention between every pixel to every other pixel.
        
        # B, C, H, W -> B, C, H*W
        x = x.view(n, c, h*w)
        x = x.transpose(-1, -2)    #B, H*W, C
        x = self.attention(x)      #B, H*W, C
        x = x.transpose(-1, -2)    #B, C, H*W
        x = x.view(n, c, h, w)     # B, C, H, W
        
        x += residue
        
        return x
        
        
        
    
            
