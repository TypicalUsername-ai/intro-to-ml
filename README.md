
# Course Outline

0. [Basics and introduction]()
1. [PyTorch fundamentals](./00_pytorch_fundamentals.ipynb)
	1. Dealing with tensors
	2. Tensor operations
2. [Preprocessing data]()
3. [Building and using pre-trained ML models]()
4. [Fitting a model to the data]()
5. [Making predictions with a model]()
6. [Evaluate model predictions]()
7. [Saving and loading models]()
8. [Using trained models to make predictions on custom data]()

# Resources

- [GitHub repo](https://github.com/mrdbourke/pytorch-deep-learning)
- [Course Q&A](https://github.com/mrdbourke/pytorch-deep-learning/discussions)
- [Course online book](https://www.learnpytorch.io)
- [PyTorch website](https://www.pytorch.org)
- [Pytorch forum](https://discuss.pytorch.org/)

# Machine learning notes

turning things (data) into numbers and finding patterns in those numbers

> [!info]
> The course is code-focused with extra math readings

## ML vs Deep learning vs traditional programming

- AI contains ML and ML contains deep learning
- traditional programming starts with the inputs and a recipe and returns the output
- an ML/DL approach contains input and output infers the recipe
- ML algorithm infers the relationships between inputs and labels
- Traditional ML models are usually used on structured data (e.g. rows and tables)
- Deep Learning is typically better for unstructured data

## Why use ML/DL

Mostly because for a complex problem it is very difficult to think of all the rules needed for the system to function properly

> If you can build a simple (or even not so very simple) rule-based system that doesn't require machine learning, do that

**ML is good for**

1. Problems with long lists of rules - e.g. driving a car as the traditional approach fails
2. Continually changing environments - Deep Learning can adapt to new scenarios
3. Discovering insights with large collections of data - it may be hard to hand-craft rules for inference of how do 101 different foods look like

**ML is (maybe) bad for**

1. When you need explainability - the patterns of a DL model are usually uninterpretable
2. When the traditional approach is a better option
3. When errors are unacceptable - since the outputs are probabilistic we reduce errors to 0
4. When you don't have much data - as the models usually require it, however there exist a few workarounds

## Common algorithms

**Machine learning**

- Random forest
- Gradient boosted models
- Naive Bayes
- Nearest neighbor
- Support vector machine

**Deep learning**

- Neural networks
- Fully connected neural network
- Convolutional neural network
- Recurrent neural network
- Transformer

## Neural networks

We have inputs which is then transformed into numbers (numerical encoding / representation), we then pass it through a neural network which then gives us an output.

> [!important]
> The neural network follows the following patterns:
> input -> encoding -> transformation -> decoding -> output

**NNs consist of the following parts**

- Input layers
- Hidden layers (learn the patterns in data)
- Output layer (predicts probabilities)

> [!info]
> Each NN layer is usually a combination of linear and non-linear functions.

## Types of learning

- Supervised learning
- Unsupervised learning
- Self-supervise learning
- Transfer learning
- Reinforcement learning

## Deep learning usages

- Recommendation
- Translation (improved)
- Speech recognition
- Computer vision (e.g. object detection)
- Natural language processing

> [!info]
> Translation and Speech recognition are sequence to sequence types
> Computer vision and NLP are classification and regression

# `PyTorch`

## What is it?

- most popular research deep learning framework (58% popularity roughly)
- Write fast deep learning code in Python
- Able to access pre-build models from e.g. torchub
- Provides a whole ecosystem stack from preprocessing to deploy
- Originally designed to use in house as Meta but now is open source

> [!important]
> [papers with code](https://www.paperswithcode.com)

## Why?

- Research favorite
- Broad functionality and ecosystem
- A lot of usages and standardization
- Majority of companies are using it
- Can be accelerated with GPU/TPU

> [!read]
> "The incredible Pytorch" GitHub repository

## Tensor

Tensor is almost any representation of numbers, most usually a multi dimensional array
Neural networks perform the mathematical operations on tensors


## Pytorch workflow

1. Get data ready
2. Build or pick a model
	1. Pick a loss function and optimizer
	2. Build a training loop
3. Fit the model to the data and make predictions
4. Evaluate the model
5. Improve through experimentation
6. Save and reload the trained model