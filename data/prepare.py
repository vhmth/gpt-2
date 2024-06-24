import os

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
        return { 'ids': tokens, 'len': len(tokens) }

    tokenized_data = split_dataset.map(
        tokenize,
        remove_columns=["text"], # remove the text columns since we'll just use the BPE ids
        desc="tokenizing the data splits",
        num_proc=num_proc
    )