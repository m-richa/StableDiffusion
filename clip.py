import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    
    def __init__(self, vocab_size, d_embed, n_tokens):
        super().__init__()
        
        #weight of embedding learned is vocab_size, d_embed
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.positional_embedding = nn.Parameter(torch.zeros((n_tokens, d_embed)))
        
    def forward(self, tokens):
        
        # W->vocab_size, d_embed; 
        #b, seq_len -> b, seq_len, d_embed
        x = self.token_embedding(tokens)
        x += self.positional_embedding
        
        return x
        
class CLIPLayer(nn.Module):
    
    def __init__(self, n_head, d_embed, ):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(n_head, d_embed)
        self.layernorm2 = nn.LayerNorm(d_embed)
        self.linear1 = nn.Linear(d_embed, 4*d_embed)
        self.linear2 = nn.Linear(4*d_embed, d_embed)
        
    def forward(self, x):
        
        #x: b, seq_len, d_embed=768
        
        residual = x
        
        x = self.layernorm1(x)
        x = self.attention(x, causal_mask=True)
        x+= residual
        
        residual = x
        x = self.layernorm2(x)
        x = self.linear2(x)
        x *= torch.sigmoid(1.702*x) #QuickGELU
        x += residual
        
        return x
        

class CLIP(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # CLIP embeddings is 768 for every token
        
        # Same as in stable diffusion
        self.embedding = CLIPEmbedding(49408, 768, 77) #Vocabulary size, embedding size, number of tokens
        
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for _ in range(12)])
        
        self.layernorm = nn.LayerNorm(768)
        
    def forward(self, tokens):
        
        tokens = tokens.type(torch.long)
        
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
            
        output = self.layernorm(state)
        
        return output
        
        
        
        
