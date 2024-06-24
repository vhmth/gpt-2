from datasets import load_dataset

if __name__ == '__main__':
    # download the huggingface openwebtext dataset: https://skylion007.github.io/OpenWebTextCorpus/
    # docs: https://huggingface.co/docs/hub/en/datasets-downloading
    dataset = load_dataset("openwebtext", trust_remote_code=True)