import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

class GPTConfig:
    block_size: int=1024
    vocab_size: int=50304
    n_layer: int=12
    n_head: int=12
    n_embd: int=768
    dropout: float=0.1
    bias: bool=True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.embd, 3*config.embd)
        self.c_proj = nn.Linear(config.embd, config.embd)
        
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        
    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        
        attn = (q@k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        
        attn = attn.masked_fill(self.bias[:,:,:T,:T], float("-inf"))
        
        attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        y = attn@v
        
        y = y.transpose(1, 2).contiguous.view(B, T, C)
        
        y = self.resid_drop(self.c_proj(y))
        
        return y
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        
class Block(nn.Module):
    def __init__(self, config):
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x