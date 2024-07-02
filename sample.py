"""
Sample from the model. If a trained model checkpoint exists, this script
will automatically load it into the model.
"""

import os

import tiktoken
import torch
import torch.nn as nn

from model import GPTConfig, GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt_config = GPTConfig(device=device)
model = nn.DataParallel(GPT(gpt_config))
model.to(device)

chkpt_file = os.path.join('out', 'chckpt.pt')

checkpoint = None
if os.path.exists(chkpt_file):
    print(f"loading checkpoint from {chkpt_file}")
    checkpoint = torch.load(chkpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    curr_epoch = checkpoint['curr_epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"loaded checkpoint, trained over {curr_epoch} epochs with best val loss: {best_val_loss}")
else:
    print("no checkpoint file found, loading untrained model")

# generate from the model
enc = tiktoken.get_encoding("gpt2")
context = torch.zeros((1,1), dtype=torch.long, device=device)

with torch.no_grad():
    print(enc.decode(model.module.generate(context, max_new_tokens=500)[0].tolist()))