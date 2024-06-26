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

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        same as usual causal self-attention but with:
        - attention Dropout
        - Linear projection for the residual network
        - residual dropout
        """

    def forward(self, x):
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        """"
        [
            # layers for the MLP
            Linear,
            GELU,

            # residual projection layer
            Linear,

            # dropout for the residual layer (in GPT-1 paper)
            Dropout
        ]
        additional for residual network: [projection,]
        Linear projection (residual network) -> Linear (learning the )
        """
        self.config = config

    def forward(self, x):
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
        pos_embd = self.position_embedding_table(torch.arange(T)) # (T, C)
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
            print(f"logits:{logits},loss:{loss}")
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)

        return idx