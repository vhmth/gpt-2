import os

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tiktoken

num_proc = os.cpu_count() // 2

if __name__ == '__main__':
    # download the huggingface openwebtext dataset: https://skylion007.github.io/OpenWebTextCorpus/
    # docs: https://huggingface.co/docs/hub/en/datasets-downloading
    dataset = load_dataset("openwebtext", trust_remote_code=True, num_proc=num_proc)

    # by default, the openwebtext dataset only has a train dataset
    # convert to train and test
    # shuffle the data
    # shuffle the data with a fixed seed so your test dataset is the same across training runs
    split_dataset = dataset["train"].train_test_split(test_size=0.005, shuffle=True, seed=1337)

    print(f"split the dataset into train and test: {split_dataset}")

    # tokenize the dataset with the gpt2 encoder
    enc = tiktoken.get_encoding("gpt2")

    def tokenize(data):
        tokens = enc.encode_ordinary(data["text"])
        tokens.append(enc.eot_token)

        # save both the ids and length so we can efficiently sum
        return { 'ids': tokens, 'len': len(tokens) }

    tokenized_data = split_dataset.map(
        tokenize,
        remove_columns=["text"], # remove the text columns since we'll just use the BPE ids
        desc="tokenizing the data splits",
        num_proc=num_proc
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    # this part was copied from nanogpt: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
    # I obviously need to get more comfortable with numpy
    for split, dset in tokenized_data.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()