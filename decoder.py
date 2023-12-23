import torch
import torch.nn as nn
from torch.nn import functional as F

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
    
class Decoder(nn.Sequential):
    
    def __init__(self, ):
        super().__init__(
            
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            #b,4,h,w -> b,512,h,w
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            ResidualBlock(512, 512),
            AttentionBlock(512),
            
            #b,512,h/8,w/8 -> b,512,h/8,w/8
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            
            #b,512,h/8,w/8 -> b,512,h/4,w/4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #b,512,h/4,w/4 -> b,512,h/4,w/4
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            
            #b,512,h/4,w/4 -> b,512,h/2,w/2
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #b,512,h/2,w/2 -> b,256,h/2,w/2
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            
            #b,256,h/2,w/2 -> b,256,h,w
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #b,256,h/2,w/2 -> b,128,h,w
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
            
        )
        
    def forward(self, x):
        #x: b,4,h/8,w/8
        
        x/= 0.18215
        
        for module in self:
            x = module(x)
            
        return x
        
        
        
    
            
