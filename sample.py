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

# options

# by default, load with the prompt "Hello, I'm a language model,"
# if this is True, loads with 0s
load_with_no_prompt = False
# -------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt_config = GPTConfig(device=device)
model = GPT(gpt_config)
model.eval()
model.to(device)

chkpt_file = os.path.join('out', 'chckpt.pt')

checkpoint = get_and_load_committed_checkpoint(model, device=device)

if checkpoint == None:
    print("no checkpoint file found, loading untrained model")

# generate from the model
enc = tiktoken.get_encoding("gpt2")

if load_with_no_prompt:
    context = torch.zeros((1,1), dtype=torch.long, device=device)
else:
    start_ids = enc.encode("Hello, I'm a language model,", allowed_special={"<|endoftext|>"})
    context = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    print(enc.decode(model.generate(context, max_new_tokens=100)[0].tolist()))