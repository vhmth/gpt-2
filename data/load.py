import os

import numpy as np
import tiktoken

"""
You must run ./prepare.py before leveraging these utilities to
download, split, tokenize, and save the training and test data.
"""

enc = tiktoken.get_encoding("gpt2")
train_data_bin = os.path.join('data', 'train.bin')
test_data_bin = os.path.join('data', 'test.bin')

# filename is a path object to either the train or test data bins
# num_docs is the number of documents you want to read from the file
def load_data(filename, num_docs = 1):
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