import os
import psutil
from psutil._common import bytes2human

import numpy as np
import tiktoken
import torch

"""
You must run ./prepare.py before leveraging these utilities to
download, split, tokenize, and save the training and test data.
"""

enc = tiktoken.get_encoding("gpt2")
train_data_bin = os.path.join('data', 'train.bin')
val_data_bin = os.path.join('data', 'val.bin')

# filename is a path object to either the train or test data bins
# num_docs is the number of documents you want to read from the file
def load_data_docs(filename, num_docs = 1):
    docs = [] # array of docs in token format
    current_doc = np.zeros((0,), np.uint16)
    data_offset = 0 # where we are in the file
    data_read_count = 1_000_000 # how many tokens to read at a time

    # note each item has a size of 2 bytes (16-bits)
    file_stats = os.stat(filename)
    file_size = file_stats.st_size
    total_items = file_size / 2

    data_arr = np.memmap(filename, dtype=np.uint16, mode='r')

    while data_offset < total_items and len(docs) < num_docs:
        data_arr = np.memmap(filename, dtype=np.uint16, mode='r')
        data = data_arr[data_offset:data_offset + data_read_count]
        data_offset += len(data)
        for id in data:
            current_doc = np.append(current_doc, id)
            if id == enc.eot_token:
                docs.append(current_doc)
                current_doc = np.zeros((0,), np.uint16)
            if len(docs) == num_docs:
                break
    
    return docs

train_data_arr = None
val_data_arr = None
train_data_size = os.stat(train_data_bin).st_size
val_data_size = os.stat(val_data_bin).st_size
total_data_arr_size = train_data_size + val_data_size
should_load_into_mem = psutil.virtual_memory().available > (total_data_arr_size + 1e9) # 1e9 (1GB) memory for good measure
def get_data_file(split = "train"):
    global val_data_arr
    global train_data_arr
    if not should_load_into_mem:
        return np.memmap(train_data_bin if split == "train" else val_data_bin, dtype=np.uint16, mode='r')

    if train_data_arr is None:
        print(f"loading {bytes2human(train_data_size)} of training data into memory (this might take a while)")
        train_data_arr = np.fromfile(train_data_bin, dtype=np.uint16)

    if val_data_arr is None:
        print(f"loading {bytes2human(val_data_size)} of val data into memory")
        val_data_arr = np.fromfile(val_data_bin, dtype=np.uint16)

    return train_data_arr if split == "train" else val_data_arr

# split is either "train" or "val"
# block_size is the context length feeding into the transformer
# batch_size is the number of examples to pull
def get_data_batch(split, block_size, batch_size, device = "cpu"):
    data_arr = get_data_file(split)
    batch_offsets = torch.randint(len(data_arr) - block_size, (batch_size,))
    X = torch.stack([torch.from_numpy(data_arr[i:i+block_size].astype(np.int64)).to(device) for i in batch_offsets])
    Y = torch.stack([torch.from_numpy(data_arr[i+1:i+1+block_size].astype(np.int64)).to(device) for i in batch_offsets])
    return X, Y