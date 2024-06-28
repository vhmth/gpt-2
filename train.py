"""
Train the model. Make sure to first download and prepare the training and test
data splits. You can do this by running:

# NOTE: this takes a while and downloads ~80GB of data!
$ python3 ./data/prepare.py

Then you can run this file:
$ python3 ./train.py

TODOS:

- get onto lightning studio
- run via torchrun (https://pytorch.org/docs/stable/elastic/run.html) for efficiency
- implement checkpoints via torch.save and torch.load
- log to wandb or neptune
- cuda-ify everything
"""

from dataclasses import asdict
from pprint import pprint

import torch

from model import GPTConfig, GPT
from data.load import get_data_batch, train_data_bin, val_data_bin

# hyperparameters
batch_size = 512
max_iters = 5000
eval_interval = 500
learning_rate = 2.5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# -------------

# load the model
gpt_config = GPTConfig()
print("initializing GPT with config:")
pprint(asdict(gpt_config), sort_dicts=False)
model = GPT(gpt_config)

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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        bin_file = train_data_bin if split == "train" else val_data_bin
        for k in range(eval_iters):
            x, y = get_data_batch(bin_file, gpt_config.block_size, batch_size)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_data_batch(train_data_bin, gpt_config.block_size, batch_size)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()