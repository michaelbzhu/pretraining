# Pretraining a transformer in 600 lines

### tl;dr
This script implements the transformer architecture, data loading, distributed training, and checkpointer logic in less than 600 lines of python code.

I ran this script on a 8xH100 80GB node for a 3.2B param model and a 7.6B param model. The weights and evaluation results are linked below:
- 3.2B: https://huggingface.co/michaelbzhu/test-3.2B-base
- 7.6B: https://huggingface.co/michaelbzhu/test-7.6B-base

### Model Architecture
Implements a dense transformer architecture with the following details:
- Feed forward network: Linear -> ReLU -> Linear
- Pre normalization: apply RMSNorm on hidden states before passing them into attn and ffn layers
- RoPE positional embeddings
- Uses GPT2 tokenizer


### Distributed (Fully Sharded Data Parallel)
We implement distributed data loading logic to send a different batch of data to each GPU. It uses the rank of the process to determine which batch of data to load.

We use `fully_shard()` from `torch.distributed.fsdp` to implement fully sharded data parallelism. This shards the model across the 8 GPUs and handles the communication needed to synchronize gradients and weights during forward and backward passes.

### Data
Dataset: [fineweb pretokenized using GPT2](https://huggingface.co/datasets/kjj0/fineweb100B-gpt2)

The data loading code was adapted from [nanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt)
