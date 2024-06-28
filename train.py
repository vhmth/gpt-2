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

import torch
from model import GPTConfig, GPT

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# -------------

# 1. load the model (TODO: pull from checkpoint)
# 2. create optimizer and establish learning loop
# 3. forward: pull out minibatches of data
# 4. zerograd
# 5. train
# 6. calculate loss
# 7. update each parameter by the learning rate (TODO: get parameters as list from model)