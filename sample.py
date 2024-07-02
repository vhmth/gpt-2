"""
Sample from the model. If a trained model checkpoint exists, this script
will automatically load it into the model.
"""

import os

import tiktoken
import torch
import torch.nn as nn

from checkpoint.checkpoint import get_and_load_committed_checkpoint
from model import GPTConfig, GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt_config = GPTConfig(device=device)
model = nn.DataParallel(GPT(gpt_config))
model.to(device)

chkpt_file = os.path.join('out', 'chckpt.pt')

checkpoint = get_and_load_committed_checkpoint(model, device=device)

if checkpoint == None:
    print("no checkpoint file found, loading untrained model")

# generate from the model
enc = tiktoken.get_encoding("gpt2")
context = torch.zeros((1,1), dtype=torch.long, device=device)

with torch.no_grad():
    print(enc.decode(model.module.generate(context, max_new_tokens=500)[0].tolist()))