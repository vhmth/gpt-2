"""
Train the model. Make sure to first download and prepare the training and test
data splits. You can do this by running:

# NOTE: this takes a while and downloads ~80GB of data!
$ python3 ./data/prepare.py

Then you can run this file:
$ python3 ./train.py

TODOS:

- log to wandb or neptune
- lightning litgpt optimizations
"""

import math
import os
import time
import sys
from dataclasses import asdict
from pprint import pprint

import torch
import torch.nn as nn
from tqdm import tqdm

from checkpoint.checkpoint import get_and_load_training_checkpoint, training_checkpoint
from model import GPTConfig, GPT
from data.load import get_data_batch

# hyperparameters
batch_size = 12
max_iters = 600000 # maximum number of training iters

should_estimate_loss = True # useful to turn this off if optimizing or debugging the main training loop
eval_interval = 1000 # every num training iters we print estimated loss at
eval_iters = 200 # number of training and val data samples we estimate loss over

warmup_iters = 2000 # linear warmup steps per the paper
learning_rate = 6e-4 # max learning rate
min_lr = 6e-5
lr_decay_iters = 600000
decay_lr = True # whether to decay the learning rate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
init_from = "scratch"
out_dir = 'out'
chkpt_file = training_checkpoint

# whether to leverage torch.compile
# make sure we can only run this on versions of python < 3.12:
# https://github.com/pytorch/pytorch/issues/120233
# compile = sys.version_info[0] < 3 or sys.version_info[1] < 12

# I would love to turn this on, but compilation throws errors with weight sharing
# and weight sharing leads to much better perf than compilation
compile = False
# -------------

# for GPU hardware that can support it, set matmul precision to "high"
# so we can take advantage of lower-precision tf32 for more throughput
torch.set_float32_matmul_precision('high')

# load the model
gpt_config = GPTConfig(device=device)
pprint(asdict(gpt_config), sort_dicts=False)
model = nn.DataParallel(GPT(gpt_config))
model.to(device)

# create optimizer
# according to the GPT-1 paper:
# - use the Adam optimizer
# - learning rate of 2.5e-4
# - increase rate linearly from 0 for first 2000 updates
# - annealed to 0 using cosine schedule
#
# check out `get_lr` below for points 2-3.
#
# we are going to use the AdamW optimizer to avoid needing to
# include the "modified L2 regularization" mentioned in the
# GPT-1 paper.
#
# more on AdamW here: https://arxiv.org/pdf/1711.05101
parameters = [p for p in list(model.parameters()) if p.requires_grad is True]
num_params = sum(p.numel() for p in parameters)
print(f"number of parameters: {num_params}")
optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

# handle checkpoints - for more info:
# https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
os.makedirs(out_dir, exist_ok=True)

if init_from == "scratch" and os.path.exists(chkpt_file):
    replace_checkpoint = input(f"""
        Checkpoint file exists at {chkpt_file}.
        Do you want to train the model from scratch and replace the checkpoint?
        If you do, respond with "y" or "yes":
    """)

    if replace_checkpoint.lower() not in ["y", "yes"]:
        resume_from_checkpoint = input(f"""
            Would you like to resume training from {chkpt_file}?
            If you do, respond with "y" or "yes":
        """)

        if replace_checkpoint.lower() in ["y", "yes"]:
            init_from = "resume"
        else:
            sys.exit()

curr_epoch = 0
best_val_loss = None
if init_from == "resume":
    checkpoint = get_and_load_training_checkpoint(model, optimizer, device)
    curr_epoch = checkpoint['curr_epoch']
    best_val_loss = checkpoint['best_val_loss']

if compile:
    print("compiling model")
    model = torch.compile(model)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters), desc=f"estimating {split} loss over {eval_iters} iters", leave=False):
            x, y = get_data_batch(split, gpt_config.block_size, batch_size, device)
            logits, loss = model(x, y)
            # must use loss.mean() in case this is returning multiple
            # losses per GPU data batch
            losses[k] = loss.mean().item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
for iter in range(curr_epoch, max_iters):
    # determine and set the learning rate for this iteration
    lr = get_lr(iter) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # every once in a while evaluate the loss on train and val sets
    if should_estimate_loss and iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr}")

        best_val_loss = losses['val'] if best_val_loss is None else min(losses['val'], best_val_loss)
        curr_epoch = iter
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'curr_epoch': curr_epoch
        }
        print(f"saving checkpoint to {chkpt_file}...")
        torch.save(checkpoint, chkpt_file)

    t0 = time.time()

    # sample a batch of data
    xb, yb = get_data_batch("train", gpt_config.block_size, batch_size, device)

    optimizer.zero_grad(set_to_none=True)

    # used mixed precision autocasting to speed up GPU throughput
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # evaluate the loss
        logits, loss = model(xb, yb)
        # must use loss.mean() in case this is returning multiple
        # losses per GPU data batch
        loss = loss.mean()

    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) * 1000
    tokens_per_sec = batch_size * gpt_config.block_size / dt
    print(f"step {iter}, loss: {loss.item()}, dt: {dt}ms, tok/sec: {tokens_per_sec}")