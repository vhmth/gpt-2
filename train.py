"""
Train the model. Make sure to first download and prepare the training and test
data splits. You can do this by running:

# NOTE: this takes a while and downloads ~80GB of data!
$ python3 ./data/prepare.py

Then you can run this file:
$ python3 ./train.py

TODOS:

- run via torchrun (https://pytorch.org/docs/stable/elastic/run.html) for efficiency
- log to wandb or neptune
- lightning litgpt optimizations
"""

import os
import sys
from dataclasses import asdict
from pprint import pprint

import torch
from tqdm import tqdm

from model import GPTConfig, GPT
from data.load import get_data_batch, train_data_bin, val_data_bin

# hyperparameters
batch_size = 12
max_iters = 10000 # maximum number of training iters

eval_interval = 500 # every num training iters we print estimated loss at
eval_iters = 200 # number of training and val data samples we estimate loss over

learning_rate = 2.5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
init_from = "scratch" # change to "resume" if loading from a checkpoint
out_dir = 'out'
chkpt_file = os.path.join(out_dir, 'chckpt.pt')
# -------------

# load the model
gpt_config = GPTConfig(device=device)
pprint(asdict(gpt_config), sort_dicts=False)
model = GPT(gpt_config)
model.to(device)

# create optimizer
# according to the GPT-1 paper:
# - use the Adam optimizer
# - learning rate of 2.5e-4
# - increase rate linearly from 0 for first 2000 updates
# - annealed to 0 using cosine schedule
#
# we are going to see if we can ignore the cosine annealing
# and linear increase for the first 2000 updates.
#
# we are going to use the AdamW optimizer to avoid needing to
# include the "modified L2 regularization" mentioned in the
# GPT-1 paper.
#
# more on AdamW here: https://arxiv.org/pdf/1711.05101
parameters = [p for p in list(model.parameters()) if p.requires_grad is True]
print(f"number of parameters: {len(parameters)}")
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
        bin_file = train_data_bin if split == "train" else val_data_bin
        for k in tqdm(range(eval_iters), desc=f"estimating {split} loss over {eval_iters} iters", leave=False):
            x, y = get_data_batch(bin_file, gpt_config.block_size, batch_size, device)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop
for iter in tqdm(range(curr_epoch, max_iters), desc=f"training GPT {max_iters} epochs"):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

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
    xb, yb = get_data_batch(train_data_bin, gpt_config.block_size, batch_size, device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()