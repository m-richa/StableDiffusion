import torch
from torch import nn
from torch.nn import functional as F
from decoder import AttentionBlock, ResidualBlock

#Reduces the dim of data but increases the number of features
class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            
            #--------------------------------------------
            #B,3,H,W -> B,128,H,W
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            #B,128,H,W -> B,128,H,W
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            #--------------------------------------------
            
            #B,128,H,W -> B,128,H/2,W/2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            #B,128,H/2,W/2 -> B,256,H/2,W/2
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            
            #B,128,H/2,W/2 -> B,256,H/4,W/4
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            #B,256,H/2,W/2 -> B,512,H/4,W/4
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            
            #--------------------------------------------
            #B,512,H/4,W/4 -> B,512,H/8,W/8
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            #--------------------------------------------
            
            ResidualBlock(512, 512),
            ##B,512,H/8,W/8 -> B,512,H/8,W/8
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            
            #B,512,H/8,W/8 -> B,8,H/8,W/8
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=1)
            
            
        )
        
    def forward(self, x, noise):
        
        # x: B, c, H, W
        # Noise: B, out_c, H/8, W/8
        
        for module in self:
            
            # when stride=2, we need to pad one more pixel on the right/bottom
            if getattr(module, 'stride', None) == (2,2):
                x = F.pad(x, (0,1,0,1))
            x = module(x)
        
        #B, 8, H/8, W/8 -> B, 4, H/8, W/8    
        mean, log_var = torch.chunk(x, 2, dim=1)
        
        var = torch.exp(torch.clamp(log_var, -20, 20))
        
        std = torch.sqrt(var)
        
        # Reparameterization trick for sampling from the above distribution
        z = mean + std * noise
        
        #scale the output????????
        x *= 0.18215
        
        return x