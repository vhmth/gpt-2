"""
Model definition. To use, check out train.py and sample.py. I have written
these files without looking at NanoGPT, but I did try to follow the same
class interfaces. Most notably, GPTConfig and GPT are imported from this
module and then used to train and sample from the trained model.
"""

from dataclasses import dataclass
import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

# Copied CausalSelfAttention from nanogpt for its speed. Got this working with
# my previous bigram implementation from the course, but it was too slow since:
#
# 1. It used separate k,v,q tensors for each head instead of 1
# 2. It did not take advantage of flash attention (scaled_dot_product_attention)
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.GPT2_SCALE_INIT = 1

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

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
        self.proj_linear.GPT2_SCALE_INIT = 1
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
    # load vocab_size to nearest divisibility of 2 to speed up token throughput on GPU
    # actual vocab_size is 50257
    vocab_size: int = 50304
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
        self.n_layer = n_layer = config.n_layer
        dropout = config.dropout
        bias = config.bias

        self.device = config.device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=bias) # final layer norm

        # normalize weights with a std of 0.02 (per gpt-2 paper)
        # note that this is roughly (but not exactly) equivalent to initializing
        # to the sqrt(fan_in)
        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            # scale down the std by 1/sqrt(N) where N is the number of layers
            # deep of residual layers in order to compensate for growing std
            # of the residual pathway
            #
            # note that we multiply by 2 here since there are 2 residual pathways
            # per block (attention and ffwd)
            if hasattr(module, 'GPT2_SCALE_INIT'):
                std *= (2 * math.sqrt(self.n_layer))**-0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = self.embedding_dropout(tok_emb + pos_embd) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)

        # weight sharing between the position embedding and language model head
        # ^ used in attention is all you need paper (section 3.4)
        #
        # this is equivalent to lm_head
        logits = F.linear(x, weight=self.token_embedding_table.weight) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # conform to what pytorch expects the matrix dims to be
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that are 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8)

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