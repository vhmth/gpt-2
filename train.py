"""
Train the model. Make sure to first download and prepare the training and test
data splits. You can do this by running:

# NOTE: this takes a while and downloads ~80GB of data!
$ python3 ./data/prepare.py

Then you can run this file:
$ python3 ./train.py

TODOS:

- get onto lightning studio
- run via torchrun (https://pytorch.org/docs/stable/elastic/run.html) for efficiency
- implement checkpoints via torch.save and torch.load
- log to wandb or neptune
- cuda-ify everything
"""