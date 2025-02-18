---
title: Artificial Intelligence in Mathematical Engineering 
subtitle: Spring 2025 (MYZ 309E)
author: Atabey Kaygun
date: Tuesday, February 18, 2025
incremental: true
---

## Course overview

### Who am I?

Originally theoretical math (algebra) turned data analyst by demand

- statical data analysis
- time series analysis
- text analysis
- topological data analysis

### Overview of the Syllabus

### What you must do

- Setup your environment: install
  + python (anaconda / pip / uv)
  + jupyter 
  + git 
- Open a github account
- Create a *private* repository for this class

### Assessment

- 5 homework projects (deadlines are on github)
- final proposal (minimum 2000 words)
- final project (mimimum 5000 words)

### Homeworks

- 4 or 5 questions each
- involves what we learned up until that point
- you have 1 week to complete
- submit on github

### The Final Proposal

You **must** talk to me before you design your project.

- minimum of 2000 words (~4 pages)
- dataset(s) with detailed description
- questions you ask the dataset(s)
- methods you are going to use

### The Final Project

- minimum 5000 words (~10 pages)
- dataset(s)
- hardware and software
- question(s)
- methods to extract answers from the dataset(s)
- obstacles and solutions
- analysis (what, why, how)

### Use of LLMs

- Transparency and maintaining academic integrity.
- I allow and even encourage students to large language models (ChatGPT, Claude2, LLAMA3 etc).
- Your required to document the usage through logged transcripts.

## The Tools

### Package Managers

Package manager manages the libraries installed on your system:

- [anaconda](https://docs.conda.io/en/latest/)
- [pip](https://pypi.org/project/pip/)

### Online Compute Platforms

* [CoCalc](https://cocalc.com/) 
* [Google Colab](https://colab.research.google.com/)
* [Microsoft Azure Notebooks](https://visualstudio.microsoft.com/vs/features/notebooks-at-microsoft/)
* [Kaggle](https://www.kaggle.com/)

### Git

[Git](https://en.wikipedia.org/wiki/Git) is a version control
system. [(Image source)](https://github.com/crc8/GitVersionTree)

<img src="../images/git-branches-merge.png" height="450">

### GitHub

[GitHub](https://github.com) is a sharing platform. [(Image source)](https://support.nesi.org.nz/hc/en-gb/articles/360001508515-Git-Reference-Sheet)

![](../images/git.svg)

## Jupyter and Markdown

### Markdown

- The most important part of your HWs, proposal, and the project are the text! 
- Text in jupyter notebooks need to be written in [Markdown](https://en.wikipedia.org/wiki/Markdown).
- Markdown is a markup language like HTML, but [much simpler](https://www.markdownguide.org/cheat-sheet/).
- Learn markdown.

### Jupyter Notebooks

- You must submit your HWs, Project Proposal and Project as a jupyter notebook!
- I will accept no other forms!

### Jupyter Notebooks

[Jupyter notebooks](https://jupyter.org/) are ideal for constructing a
coherent narrative analysis of data because

- Interactive.
- Versatile (many languages) [JU](https://julialang.org/)(lia)[PYT](https://www.python.org/)(hon)e[R](https://www.r-project.org/).
- Supports code, text, and visualization in the same context.

## Languages

### Python

[Python](https://python.org) is 

- dynamically type-checked 
- garbage-collected
- object-oriented 
- functional

### Python 

- Popular open source programming language
- Flexible language good for beginners
- Widely used in industry and research

### Library Ecosystem

Many libraries for data analysis 

+ [Pandas](https://pandas.pydata.org/), [Polars](https://pola.rs/), [DuckDB](https://duckdb.org/)
+ [Numpy](https://numpy.org/), [JAX](https://docs.jax.dev/en/latest/quickstart.html)
+ [Scikit-Learn](https://scikit-learn.org/stable/)
+ [Tensorflow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/)
+ [Networkx](https://networkx.org/)
+ [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Gravis](https://robert-haas.github.io/gravis-docs/), [Altair](https://altair-viz.github.io/)

### SQL

[SQL](https://en.wikipedia.org/wiki/SQL) is the industry standard for data filtering/merging/munging 

- Popular 
- Standard
- Simple

We will use via [DuckDB](https://duckdb.org/).

## Data

### What is Data?

Any collection of symbols can be data as long as they are

- recorded consistently
- collected within the same context
- recalled consistently and accurately

### Structured vs Unstructured data

Data can be 

- Structured
- Unstructured

### Structured data

- *Structure* refers to the shape
- Data may come in 
  + [arrays](https://en.wikipedia.org/wiki/Array_(data_structure)) (rows, columns)
  + [trees](https://en.wikipedia.org/wiki/Tree_(abstract_data_type))
  
### Unstructured data

- text
- images
- ???

### Data types

Data type refers to the elements of a data heap

- discrete numerical
- continuous numerical
- ordered categorical
- unordered categorical
- univariate
- multivariate

### Format 

*Format* refers how data is encoded

- Columnar data
  + [Comma Separated Vectors](https://en.wikipedia.org/wiki/Comma-separated_values) (CSV)
  + [Excel](https://en.wikipedia.org/wiki/Microsoft_Excel#File_formats)
  + [Parquet](https://parquet.apache.org/)
  + [HDF](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)

- Tree data
  + [JSON](https://en.wikipedia.org/wiki/JSON)
  + [XML](https://en.wikipedia.org/wiki/XML)
  + [YAML](https://en.wikipedia.org/wiki/YAML)

### Data storage

Data may be stored in

- local file system or data servers
- remote file system or data servers 

### Data access

- file read ([Pandas](https://pandas.pydata.org/), [Polars](https://pola.rs/), [DuckDB](https://duckdb.org/), [Sqlite](https://www.sqlite.org/))
- server connection (SQL or API)


## Data Science

### What is Data Science?

- interdisciplinary field 
- extracts insights from data
- fits models on data

### What is a Model?

>  "All models are wrong, but some are useful." George Box

+ Data is usually noisy, complex and hard to understand.
+ Model is an artificial construct.
+ Model is a simplification.

### What does a model do?

- Models can be 
  + predictive 
  + or descriptive.

### Data Science Workflow

<img src="../images/datascience-workflow.png" height="400">

### Analysis Workflow

1. Look at the data, clean it, understand it.
2. Put forth a hypothesis.
3. Design a question to test the hypothesis.
4. Choose a model type.
3. Design a fit function.
4. Find the best fitting model parameters.
3. Validate the model using the fit function.
4. If necessary go back to step 2, 3 or 4.
5. Write a report.

### How is it different than Science?

In any science we do the same.

+ Our domain is fixed.
+ We need deep domain knowledge.
+ Our models are specific to that particular domain.

### How is Data Analysis different than Science?

* Data Analysis models use

  + internal structure of the data
  + are domain agnostic

* Developing these models requires
  - statistical
  - mathematical
  - computational skills
  
### Do we need all of this to do Data Analysis?

* No code no compute!
* No statistics no certainity!
* No mathematics no maintanence!

![I fixed it!](../images/whac-a-mole.gif)

### Can anyone do Data Analysis without domain knowledge?

Any fool can develop a model! (coding) 

![](../images/200w.gif)

### Can anyone do Data Analysis without domain knowledge?

It is difficult to see/decide if

* a model fits (domain knowledge)
* when/where model stops working (math/stats)
* fixing a model when it breaks (math)


