"""
Model definition. To use, check out train.py and sample.py. I have written
these files without looking at NanoGPT, but I did try to follow the same
class interfaces. Most notably, GPTConfig and GPT are imported from this
module and then used to train and sample from the trained model.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

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

class GPT(nn.module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self):
        return None, None # logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        return idx