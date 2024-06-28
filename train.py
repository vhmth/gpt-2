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

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
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

# establish learning loop
# forward: pull out minibatches of data
# zerograd
# train
# calculate loss
# update each parameter by the learning rate (TODO: get parameters as list from model)