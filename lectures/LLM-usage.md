---
title: Use of LLMs in MYZ309E
subtitle: Tips, pitfalls, do's and dont's
author: Atabey Kaygun
date: Tuesday, February 25, 2025
incremental: true
---

# Use of LLMs

## What is a Large Language Model

- [Large Language Models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model) are a class
  machine learning architectures designed for [natural language processing
  (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing) tasks.

- LLMs are based on [recurrent neural networks
  (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network) that are designed to process
  sequential data.
  
- What makes them different is the [attentian
  mechanism](https://en.wikipedia.org/wiki/Attention_(machine_learning))

## LLM components

- **An Encoder**: converting input text to vectors
- **Self-Attention Mechanism**: 
    + weigh the relative importance tokens in the input sequence 
	+ produce a weighted sum of these tokens
- **Decoder**: produces a sequence of tokens that represent the context

## Training an LLM

- typically trained using masked language modeling
- predicting missing tokens in a given sequence of tokens
- in training some tokens are randomly removed (masked) 
- model predicts the missing token
- repeated many times to learn patterns 

# Writing Prompts

## What is a prompt?

- a piece of text to guide model's response
- provides the input for the model that describes
  + the context 
  + the task
  + the output
- output based on its internal knowledge

## Constructing Prompts

## Provide a context

- be clear and concise
- use a simple language
- use specific keywords

## Specify output

- clearly describe what you want
  + answer
  + explanation
  + comparison
- phrase questions in a neutral way
- provide an example if possible

## An Example

[Here](./example-1.html)
