"""
Model definition. To use, check out train.py and sample.py. I have written
these files without looking at NanoGPT, but I did try to follow the same
class interfaces. Most notably, GPTConfig and GPT are imported from this
module and then used to train and sample from the trained model.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x

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
        """
        GPT-1:
        [masked self-attention, layernorm, feed forward, layernorm]

        with GPT-2 modifications from paper:
        [layernorm, masked self-attention, layernorm, feed forward]
        """
        self.config = config

    def forward(self, x):
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
        self.config = config

    def forward(self):
        return None, None # logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        return idx