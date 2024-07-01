"""
Model definition. To use, check out train.py and sample.py. I have written
these files without looking at NanoGPT, but I did try to follow the same
class interfaces. Most notably, GPTConfig and GPT are imported from this
module and then used to train and sample from the trained model.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# modified from previous lesson with bigram model
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config, head_size):
        super().__init__()
        n_embd = config.n_embd
        bias = config.bias
        block_size = config.block_size

        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v
        return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_head = config.n_head
        n_embd = config.n_embd
        self.heads = nn.ModuleList([Head(config, n_embd // n_head) for _ in range(n_head)])
        self.residual_proj = nn.Linear(n_embd, n_embd) # projection layer going back into the residual pathway
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.residual_dropout(self.residual_proj(out))
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        bias = config.bias
        self.ln1 = nn.LayerNorm(n_embd, bias=bias)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        # note: "x +" represents a residual connection
        # you will need projection layers in the attention
        # and ffwd blocks to learn whether this identity
        # flow-through gradient is better in the context of
        # the training data!
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd

        # core ffwd
        self.ffwd_linear = nn.Linear(n_embd, 4*n_embd, bias=config.bias) # 4 * per GPT-1 paper specs of inner dimension
        self.gelu = nn.GELU()

        # learned residual projection
        self.proj_linear = nn.Linear(4*n_embd, n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.ffwd_linear(x)
        x = self.gelu(x)
        x = self.proj_linear(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        bias = config.bias
        self.ln1 = nn.LayerNorm(n_embd, bias=bias)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# largely copied from nanogpt
@dataclass
class GPTConfig:
    device: str = "cpu" # cpu or cuda
    block_size: int = 1024 # context length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        block_size = config.block_size
        vocab_size = config.vocab_size
        n_layer = config.n_layer
        dropout = config.dropout
        bias = config.bias

        self.device = config.device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.ln_f = nn.LayerNorm(n_embd, bias=bias) # final layer norm

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = self.embedding_dropout(tok_emb + pos_embd) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # conform to what pytorch expects the matrix dims to be
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)

        return idx