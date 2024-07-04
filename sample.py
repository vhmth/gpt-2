"""
Sample from the model. If a trained model checkpoint exists, this script
will automatically load it into the model.
"""

import argparse
import os

import tiktoken
import torch
import torch.nn as nn

from checkpoint.checkpoint import get_and_load_committed_checkpoint
from model import GPTConfig, GPT

default_prompt = ""
default_max_tokens = 100

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", help="Provide a prompt for the model to complete")
parser.add_argument("-m", "--max-tokens", help="The maximum number of new tokens to generate from the model")

args = parser.parse_args()
prompt = args.prompt.strip() if args.prompt is not None else default_prompt
max_new_tokens = int(args.max_tokens.strip()) if args.max_tokens is not None else default_max_tokens

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")

if prompt == "":
    print(f"no prompt provided, generating from model with no prompt with {max_new_tokens} max tokens")
    context = torch.zeros((1,1), dtype=torch.long, device=device)
else:
    print(f"generating from model with prompt \"{prompt}\" with {max_new_tokens} max tokens")
    start_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    context = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

gpt_config = GPTConfig(device=device)
model = GPT(gpt_config)
model.eval()
model.to(device)
chkpt_file = os.path.join('out', 'chckpt.pt')
checkpoint = get_and_load_committed_checkpoint(model, device=device, unwanted_prefix='module._orig_mod.')

if checkpoint == None:
    print("no checkpoint file found, loading untrained model")

with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        print(enc.decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))