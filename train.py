"""
Train the model. Make sure to first download and prepare the training and test
data splits. You can do this by running:

# NOTE: this takes a while and downloads ~80GB of data!
$ python3 ./data/prepare.py

Then you can run this file (for a single GPU or CPU):
$ python3 ./train.py

Or with DDP on 4 gpus on 1 node (for example):
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import math
import os
import time
import sys
from dataclasses import asdict
from pprint import pprint

import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from tqdm import tqdm

from checkpoint.checkpoint import get_and_load_training_checkpoint, training_checkpoint
from model import GPTConfig, GPT
from data.load import get_data_batch

# hyperparameters
batch_size = 16 # micro batch size
max_iters = 600000 # maximum number of training iters

should_estimate_loss = True # useful to turn this off if optimizing or debugging the main training loop
eval_interval = 1000 # every num training iters we print estimated loss at
eval_iters = 200 # number of training and val data samples we estimate loss over

warmup_iters = 2000 # linear warmup steps per the paper
learning_rate = 6e-4 # max learning rate
min_lr = 6e-5 # (10% of max learning rate per GPT-3 paper)
lr_decay_iters = 600000
decay_lr = True # whether to decay the learning rate
adamw_betas = (0.9, 0.95) # from GPT-3

use_checkpoint = True
init_from = "scratch" # change to "resume" to resume training from a checkpoint
out_dir = 'out'
chkpt_file = training_checkpoint

# whether to leverage torch.compile
# make sure we can only run this on versions of python < 3.12:
# https://github.com/pytorch/pytorch/issues/120233
compile = sys.version_info[0] < 3 or sys.version_info[1] < 12
# -------------

# set up DDP (distributed data parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of ddp demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    device_type = 'cuda'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing, etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to auto-detect
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = device
    print(f"using device: {device}")

def print_if_master(arg):
    if master_process:
        print(arg)

# for GPU hardware that can support it, set matmul precision to "high"
# so we can take advantage of lower-precision tf32 for more throughput
torch.set_float32_matmul_precision('high')

# load the model
gpt_config = GPTConfig(device=device)
block_size = gpt_config.block_size
if master_process:
    pprint(asdict(gpt_config), sort_dicts=False)

model = GPT(gpt_config)
model.to(device)
if compile:
    print("compiling model")
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# gradient accumulation calculations
total_batch_size = 524288 # 2^19, ~0.5M tokens (following GPT-3 paper)
assert total_batch_size % (batch_size * block_size * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (batch_size * block_size * ddp_world_size)
print_if_master(f"total desired batch size: {total_batch_size}")
print_if_master(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

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
print_if_master(f"number of parameters: {num_params}")
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, betas=adamw_betas, device=device)

# handle checkpoints - for more info:
# https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
os.makedirs(out_dir, exist_ok=True)

curr_epoch = 0
best_val_loss = None
if use_checkpoint and init_from == "resume":
    checkpoint = get_and_load_training_checkpoint(model, optimizer, device)
    curr_epoch = checkpoint['curr_epoch']
    best_val_loss = checkpoint['best_val_loss']

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        print(f"estimating {split} loss over {eval_iters} iters")
        for k in range(eval_iters):
            x, y = get_data_batch(split, block_size, batch_size, device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            losses[k] = loss.item()
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
    if should_estimate_loss and master_process and iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr}")

        best_val_loss = losses['val'] if best_val_loss is None else min(losses['val'], best_val_loss)
        curr_epoch = iter

        if use_checkpoint:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'curr_epoch': curr_epoch
            }
            print(f"saving checkpoint to {chkpt_file}...")
            torch.save(checkpoint, chkpt_file)

    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    # gradient accumulation
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        xb, yb = get_data_batch("train", block_size, batch_size, device)
        # used mixed precision autocasting to speed up GPU throughput
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # evaluate the loss
            logits, loss = model(xb, yb)
        loss = loss / grad_accum_steps # scale the loss to account for gradient accumluation
        loss_accum += loss.detach()

        # hacky way to allreduce the loss without having to duplicate
        # code and use a context manager
        if ddp:
            model.require_backward_grad_sync = (micro_step == (grad_accum_steps - 1))

        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent the model from getting gradient shocks
    optimizer.step()
    torch.cuda.synchronize() # wait for GPU to finish work before taking time
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = batch_size * block_size * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    print_if_master(f"step {iter:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()