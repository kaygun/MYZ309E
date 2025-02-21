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

- They generate text based on learned statistical distributions. 

- They function as probabilistic sequence generators based on [recurrent neural networks
  (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network)that process sequential data.

## What is a Large Language Model

- They leverage high-dimensional token embeddings and transformer architectures 

- What makes them different is the [attentian
  mechanism](https://en.wikipedia.org/wiki/Attention_(machine_learning))

## Components of an LLM

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

# Prompts

## What is a prompt?

- a piece of text to guide model's response
- provides the input for the model that describes
  + the context 
  + the task
  + the output
- output based on its internal knowledge

## Anatomy of a good prompt

- your goal(s)
- the return format
- any wanrnings
- context

## Specify your goal

- LLMs sample from probability distributions over words. 
- Ambiguity increases the entropy.
- This may result in less useful esponses.

## Goal: best practices

- State what you need.
- Use specific keywords.
- Use specific mathematical notation if applicable.
- Avoid open-ended and vague language.
- Disambiguate: "gradient" in optimization vs. vector calculus.

## Goal: an example

❌ *"Describe gradients."*  
✅ *"In the context of deep learning, describe the role of gradients in backpropagation, specifically focusing on how they relate to the Jacobian matrix of the loss function with respect to network parameters."*

## Specify the return format

- LLMs generate text autoregressively. 
- Defining structure constrains the output to an expected form.
- This improves readability and usability.

## Format: best practices

- Specify desired structure (e.g., step-by-step, theorem-proof format, code block).
- Indicate output format (e.g., LaTeX, markdown, JSON, Python).
- Clearly describe what you want (e.g. answer, explanation, comparison).
- Indicate depth (e.g., formal proof vs. intuition).

## Format: an example

❌ *"Write Python code for matrix inversion."*  
✅ *"Provide a Python implementation of matrix inversion using NumPy, formatted as a function definition, and include time complexity analysis."*

## Add constraints 

- LLMs operate on learned distributions.
- These tend to default to general assumptions. 
- Explicitly stating constraints forces model to conform.

## Constraints: best practices

- State any numerical constraints (e.g., "for n < 1000" or "in the limit as x → 0").
- Define assumptions on inputs (e.g., "Assume all functions are continuously differentiable").
- Clarify computational constraints (e.g., "Optimize for O(n log n) complexity").

## Constraints: and example

❌ *"Find the eigenvalues of a matrix."*  
✅ *"Find the eigenvalues of a symmetric 3×3 matrix with real entries and discuss their properties in terms of the spectral theorem."*

## Provide Context 

- LLMs generate responses based on conditional probabilities. 
- Without sufficient context model output distribution is broad and not relevant.

## Context: best practices

- Define domain (e.g., mathematical, scientific, programming-related).
- Provide background.
- Reiterate key points instead of assuming model retains session memory.

## Context: and example

❌ *"Explain entropy."*  
✅ *"Explain entropy in the context of information theory, assuming the Shannon definition, and provide the mathematical formulation."*


