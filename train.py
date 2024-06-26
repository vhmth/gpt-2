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
import sys
from dataclasses import asdict
from pprint import pprint

import torch
import torch.nn as nn
from tqdm import tqdm

from model import GPTConfig, GPT
from data.load import get_data_batch

# hyperparameters
batch_size = 12
max_iters = 600000 # maximum number of training iters

eval_interval = 1000 # every num training iters we print estimated loss at
eval_iters = 200 # number of training and val data samples we estimate loss over

warmup_iters = 2000 # linear warmup steps per the paper
learning_rate = 6e-4 # max learning rate
min_lr = 6e-5
lr_decay_iters = 600000
decay_lr = True # whether to decay the learning rate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
init_from = "scratch" # change to "resume" if loading from a checkpoint
out_dir = 'out'
chkpt_file = os.path.join(out_dir, 'chckpt.pt')
# -------------

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
        Are you sure you want to train the model from scratch and replace the checkpoint?
        If you don't, close out and replace `init_from = "resume"` in model.py.
        If you do, respond with "y" or "yes":
    """)

    if replace_checkpoint.lower() not in ["y", "yes"]:
        sys.exit()

curr_epoch = 0
best_val_loss = None
if init_from == "resume" and os.path.exists(chkpt_file):
    checkpoint = torch.load(chkpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    curr_epoch = checkpoint['curr_epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"loaded from checkpoint, current epoch: {curr_epoch}, best val loss: {best_val_loss}")

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
for iter in tqdm(range(curr_epoch, max_iters), desc=f"training GPT {max_iters} epochs"):
    # determine and set the learning rate for this iteration
    lr = get_lr(iter) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
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

    # sample a batch of data
    xb, yb = get_data_batch("train", gpt_config.block_size, batch_size, device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    # must use loss.mean() in case this is returning multiple
    # losses per GPU data batch
    loss = loss.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()