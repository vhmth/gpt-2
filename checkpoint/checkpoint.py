"""
Utilities to load checkpoint files. Two locations for checkpoints:

1. This folder, which is the "committed" checkpoint file uploaded
to GitHub.
2. /out/ which is the directory saved to during training. This
checkpoint is not committed to the repo but is used when resuming
training.
"""

import os

import torch

# note that the -X part of the committed checkpoint correlates
# with how many epochs we've trained the model
committed_checkpoint = os.path.join(os.path.dirname(__file__), 'chckpt-15k.pt')
training_checkpoint = os.path.join(os.path.dirname(__file__), '../out/', 'chckpt.pt')

def get_checkpoint(chkpt_file, device):
    print(f"loading checkpoint from {chkpt_file}")

    # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349
    if device == 'cpu':
        return torch.load(chkpt_file, map_location=lambda storage, loc: storage)

    # by default, we assume the checkpoint was saved from a training job on GPUs
    # and that you want to load on the same machine with GPUs
    return torch.load(chkpt_file)

def get_and_load_checkpoint(chkpt_file, model, optimizer = None, device = None, unwanted_prefix = None):
    if not os.path.exists(chkpt_file):
        return None

    checkpoint = get_checkpoint(chkpt_file, device)

    state_dict = checkpoint['model_state_dict']
    if unwanted_prefix is not None:
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    curr_epoch = checkpoint['curr_epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"loaded checkpoint {chkpt_file}, trained over {curr_epoch} epochs with best val loss: {best_val_loss}")
    
    return checkpoint

def get_and_load_committed_checkpoint(model, optimizer = None, device = None, unwanted_prefix = None):
    return get_and_load_checkpoint(committed_checkpoint, model, optimizer, device, unwanted_prefix)

def get_and_load_training_checkpoint(model, optimizer = None, device = None, unwanted_prefix = None):
    return get_and_load_checkpoint(training_checkpoint, model, optimizer, device, modify_state_dict, unwanted_prefix)