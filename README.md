# GPT-2

Recreating gpt-2 from scratch. This is from Andrej Karpathy's [final YT video](https://www.youtube.com/watch?v=l8pRSuU81PU&ab_channel=AndrejKarpathy)
in the Zero to Hero Deep Learning series. Instead of following along with the video,
I am going to attempt to build the model from scratch by only reading the GPT-2
and GPT-1 papers and figuring stuff out on my own.

## Links

- My iPad notes: https://www.icloud.com/notes/093qXDGpmFNV63nw9uSiGI6kA
- GPT-2 Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- GPT-1 Paper: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

## Data

GPT-2 was built off the WebText dataset. This dataset is internal to
OpenAI, so I will be using the [OpenWebText dataset](https://paperswithcode.com/dataset/openwebtext).
You can find all data in the [data](data/) directory.

Notably, the WebText dataset was scraped with the following constraints:

- All outbound links from Reddit posts with at least 3 karma
- All posts up until December 2017
- ~8 million documents total
- ~40GB of text
- Removal of all wikipedia documents and links since Wikipedia is
"a common data source for other datasets and could complicate the analysis due to overlapping training data with test evaluation tasks".

OpenAI did not leverage CommonCrawl to reduce the data quality complexity they would have to surmount.
Their main aim was to show that unsupervised learning on a large corpus could lead to meta learning on multiple tasks.

## Tokenization

OpenAI leveraged BPE (byte pair encoding) on top of UTF-8 unicode points
to represent the text data. They then tokenized on sub-word groupings with
a vocab size of 50,527. They leveraged other token pre-processing steps to
prevent things like BPE merging across character categories for any byte
sequence.

Since the aim of this project is to just recreate the core of GPT-2, I will
be leveraging [tiktoken](https://github.com/openai/tiktoken) instead of
implementing and training the tokenizer from scratch. This should also allow
me to download the open source weights and know that my model can interop with
whatever setup OpenAI used internally.

## Model Architecture

GPT-2 largely follows the GPT-1 architecture, which consists of:

* 12-layer decoder-only transformer
* Masked self-attention with 768 dim states and 12 attention heads
* position-wise feed-forward networks with 3072 dim inner state
* Adam optimizer with a learning rate of ~2.5e-4
* Dropout with a rate of 0.1 regularization at the residual, embedding, and attention layers
* A modified version of L2 regularization with w=0.01 on all non-bias or gain weights
* GELU for the activation functions

With some modifications:

* LayerNorm was moved to the input of each sub-block
* An additional LayerNorm was added after the final self-attention block
* Modified initialization that accounts for accumulations on the residual path with model depth
* Scaled weights of the residual layers by a factor of 1/sqrt(N) where N is the number of residual layers
* Context size of 1024
* Batch size of 512
