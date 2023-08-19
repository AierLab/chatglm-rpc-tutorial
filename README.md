# ChatGLM-RPC-Tutorial

## Overview

This repository is dedicated to implementing split learning of ChatGLM models using PyTorch RPC. The implementation is structured as a framework that can be used for our subsequent experiments in distributed learning settings.

## Tutorials

This repository is organized into a series of tutorials that progressively build up the split learning framework.

### Tutorial 1: Local Fine-Tuning of ChatGLM

This tutorial implements two ways of local fine-tuning of ChatGLM models:

1. **DeepSpeed**: An implementation of the DeepSpeed optimization. This will fine-tune the full weights of the model, and it has been tested that fine-tuning the chatGLM-6b version requires at least 30G of memory.
2. **Ptuning V2**: An alternate approach for fine-tuning.

This part of the code refers to the [ChatGLM official fine-tuning tutorial](https://www.heywhale.com/mw/project/6436d82948f7da1fee2be59e).

### Tutorial 2: Extended Functions for Experiments

Based on Tutorial 1, this tutorial further implements the necessary functions needed for various experimental setups:

1. **Layer Freezing**: Ability to freeze a specific layer while other layers participate in the training.
2. **Dynamic Layer Freezing**: Ability to change the freeze layer after a specified number of epochs.
3. **Federated Learning**: Ability to split the dataset, train it separately, and then aggregate it (federated learning).
4. **Save Intermediate Results**: Ability to save the input and output result of a specific layer during the inference.
5. **Selective Aggregation**: Keep part of the layers at the front and back ends from participating in the aggregation and output the cut layer parameters.

## Prerequisites

- Python
- transformers = 4.27.1
- PyTorch
- PyTorch RPC
- DeepSpeed

```bash
# Install depenences
pip install rouge_chinese nltk jieba datasets -i https://mirror.sjtu.edu.cn/pypi/web/simple

# Install Adgen dataset
wget -O AdvertiseGen.tar.gz https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1

# Unzip the dataset
tar -xzvf AdvertiseGen.tar.gz
```

## TODO

- [X] Solve out-of-memory issue.
- [ ] Integrate with rpc distributed learning.
