import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import torch.nn.functional as F

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    


def get_positional_encoding(max_len, d_model):
    # Initialize a tensor for positional encoding
    pe = torch.zeros(max_len, d_model)
    
    # Create a tensor representing the positions
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    # Compute the div_term using the formula (10000^(2i/d_model))
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # Apply sine to even indices
    pe[:, 0::2] = torch.sin(position * div_term)
    
    # Apply cosine to odd indices
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


class CompressionEncoder(nn.Module):
    def __init__(self, encoding_dimensionality, word_embedding_dim, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dim = dim
        self.word_embedding_dim = word_embedding_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, word_embedding_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, encoding_dimensionality)

    def forward(self, x, device):
        b, l, c = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        
        x =  x + get_positional_encoding(l, self.word_embedding_dim).to(device)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1)

        # return in classes, positions
        return self.mlp_head(x)
    


class OutputModel(nn.Module):
    def __init__(self, num_tokens, encoding_dimensionality, word_embedding_dim, dim, depth, heads, mlp_dim, window_length, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dim = dim
        self.window_length = window_length
        self.encoding_dimensionality = encoding_dimensionality

        self.embeddings = nn.Embedding(num_tokens, word_embedding_dim)
        self.projection = nn.Linear(word_embedding_dim, encoding_dimensionality) # adjusts the dimensionality so it can be attended with the encoded vectors
        self.positional_embedding_recent = nn.Parameter(torch.randn(1, window_length, word_embedding_dim)) # trained embedding of sliding window
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoding_dimensionality))
        self.offset_embedding = nn.Parameter(torch.randn(window_length + 1, encoding_dimensionality)) # index by offset
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_tokens)

    def forward(self, compressed, recent, offset, device): # offset = length % chunk_size,
        if offset is not None:
            offset = self.offset_embedding[(offset, )]
        else:
            offset = self.offset_embedding[(self.window_length, )] # offsetting by 1024 is the same as 0, technically we have 1 unnecesary dim
        offset = offset.reshape(1, 1, offset.numel()) # reshape in preperation for concatenation
        # process data to put in correct shape
        _, l_recent, _ = recent.shape
        recent = recent + self.positional_embedding_recent[:, :l_recent, :]
        recent = self.projection(recent)

        if compressed is not None:
            b, l_compressed, _ = compressed.shape
            compressed = compressed + get_positional_encoding(l_compressed, self.encoding_dimensionality).to(device)
            x = torch.cat((self.cls_token, offset, compressed, recent), dim=1)
        else:
            x = torch.cat((self.cls_token, offset, recent), dim = 1)
        
        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0, :]

        x = self.to_latent(x)
        # return in classes, positions
        return self.mlp_head(x)
    
    def embed(self, x):
        return self.embeddings(x)