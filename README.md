# Artificial Intelligence in Mathematical Engineering

This class used to be MAT388E (Data Analysis for Fundamental Sciences) and now
it has a new name. Don't ask me.

- Atabey Kaygun ([kaygun@itu.edu.tr](mailto:kaygun@itu.edu.tr))
- Lectures: Tuesdays 14:30-17:30 (D203)
 
## Course Description

This course is designed to provide a solid rigorous foundation in statistical data analysis to its
students. We will focus on both practical computational techniques, and the mathematical and
statistical theory behind these techniques. This course is for advanced undergraduate students in
mathematics who would like to gain a strong mathematical and algorithmic understanding of modern
data analysis methods.

## Books 

+ Ethem Alpaydın, "*Introduction to Machine Learning*." MIT Press.
+ Stuart J. Russell and Peter Norvig, "*Artificial Intelligence: A Modern Approach*." Prentice Hall.
+ Trevor Hastie, Robert Tibshirani, Jerome H. Friedman, "*Elements of Statistical Learning.*" Springer.
+ Kevin P. Murphy, "*Machine Learning: A Probabilistic Perspective*." MIT Press.
+ Jake VanderPlas, "*Python Data Science Handbook*." Available on
  [GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).

## Course Overview

The course will begin with classical statistical methods such as hypothesis testing and regression
models, before we progress to fundamental machine learning techniques. The course will rely heavily
on essential mathematical tools from linear algebra, probability theory, and optimization that
underlie these methods. By the end of the course, we expect students to be equipped with the
necessary skills to analyze complex datasets, apply machine learning models, and critically assess
statistical methodologies.

The course has a strong computational component. Students are expected to analyze and deploy
statistical and machine learning models using the Python language and its library ecosystem. We
will use libraries such as NumPy, pandas, duckdb, scikit-learn, TensorFlow/PyTorch, and
visualization tools such as matplotlib and seaborn. Computational assignments will involve (see
below) working with real-world datasets sourced from open data repositories. I will ask students to
apply their acquired knowledge in meaningful ways. You will use real datasets drawn from wide
variety of domains including finance, healthcare, geospatial analysis, and social sciences.

## Assessment

| **Homework**   | **Date** | **Percentage** |
|----------------|----------|----------------|
| GitHub Link    | Feb 25   | 5%             |
| Homework 1     | March 18 | 10%            |
| Homework 2     | April 7  | 10%            |
| Homework 3     | April 22 | 10%            |
| Final Proposal | April 29 | 10%            |
| Homework 4     | May 6    | 10%            |
| Homework 5     | May 20   | 10%            |
| Final Project  | June 9   | 35%            |

I am going to assign 5 homeworks.  Each of these homeworks will be published on the course github
page at

> [https://github.com/kaygun/2025-Spring-MYZ309E](https://github.com/kaygun/2025-Spring-MYZ309E) 

For each homework you are going to have 1 week to complete. Depending on your performance, I may
choose several homeworks in each turn and ask oral presentations on the howeworks handed in.

For the final project, you need to talk to me one-on-one to determine your final project to
complete depending on your interests.  Again, depending on your performance, I might also ask you
to give an oral presentation on your final project.

## Attendance

I will collect a written attendance in each lecture. I will use the attendance records for those 
students that are edge cases in their grades. (Push them up or down.)

## Technical requirements

The course is an applied data analysis class, and your performance is going to be judged from 5
homeworks, and 1 final project.  This means the course requires a degree of proficiency of
computational tools from which you are going to be responsible.  (Links provided.)

* [GitHub](https://github.com)
* [Python programming language](https://python.org)
* [Anaconda package control system](https://www.anaconda.com/products/individual)
* [Jupyter notebooks](https://jupyter.org)
* [Markdown markup language](https://www.markdownguide.org/cheat-sheet/)

Each student is going to be asked to open an [GitHub][1] account, and a **private** repository for
this class and share it with my github account at `atabey_kaygun@hotmail.com`.  You will submit
your homeworks on GitHub: I am going to pull them from your GitHub account at 11:59PM on each
deadline.  The homeworks are going to be [jupyter][2] notebooks written in [python][3]
language. You will need to install these tools on your local computational setup and learn to work
with these tools on your own. Do not ask me to help you if something does not work as there are
almost infinitely many different hardware/software setups in the wild.  If you can't install these
on your machines, you may try the following online notebook systems.

* [Google Colab](https://colab.research.google.com)
* [Microsoft Azure Notebooks](https://notebooks.azure.com/)
* [CoCalc](https://cocalc.com/)

[1]: https://github.com
[2]: https://jupyter.org
[3]: https://python.org

## Use of Large Language Models

You may use large language models (ChatGPT, Llama, Claude, Code Pilot etc.)  to assist you to code
and write your HWs. However, you must include a log of your interaction with the LLM you are using.

## Cheating, Copy/Pasting

On the other hand, passing someone else's code or text as your own without proper attribution
(including from LLMs) is cheating, or worse yet, theft. Copying code with variable names changed
from a source without proper attribution is another form of cheating. Cheaters will receive 0 and
be reported to the university. In short, don't do it.

## E-Mail Policy

I receive approximately 50 e-mails per day. So, if you need to contact me, please use the subject
``MYZ309E'' in your e-mails. Spend time in structuring your e-mail with grammatically correct
sentences in Turkish or in English. Be polite, direct, and concise. State what you need in the
first two sentences.  Sign your e-mails with your name and student number. If I can't figure out
who you are and what you need within 30 seconds of opening your message, I will delete your e-mail
with no response. You are hereby warned.

## Weekly Course Plan

Caveat emptor! The weekly plan I share here is a *plan*, and as with all plans they change. I may
go fast or slow depending on the week. I may change the order of material you see below, remove, or
add new material depending on the questions, comments, or requests.

| **Week** | **Subject**                                                                            |
|----------|----------------------------------------------------------------------------------------|
| Feb 18   | Data science, machine learning, statistics and computer science.                       |
|          | Connections, similarities, differences and interactions.                               |
| Feb 25   | A crash course in computational tools: python and ecosystem of machine learning tools. |
|          | The use of LLMs: Tips, pitfalls, do's and dont's.                                      |
|          | **Deadline for submitting GitHub links.**                                              |
| Mar 4    | Supervised vs unsupervised learning models. Models and tests.                          |
|          | Hypothesis testing. Statistical tests. Cross-validation.                               |
|          | An example: Classification vs clustering. k-means vs k-nn.                             |
| Mar 11   | Cost functions, distance functions, similarity measures.                               |
|          | Optimization and regularization.                                                       |
|          | An example: Hiearchical clustering and density based clustering.                       |
|          | **HW1 is posted.**                                                                     |
| Mar 18   | Entropy and Gini coefficient. Decision trees and random forests.                       |
|          | Assessing the quality of clusters in a clustering algorithm.                           |
|          | **Deadline for HW1.**                                                                  |
| Mar 25   | Least square regression. $R^2$, ANOVA, AIC and BIC.                                    |
|          | Regularized regressions, ridge, and lasso regression.                                  |
|          | **HW2 is posted.**                                                                     |
| Apr  7   | Logistic and multinomial regression.                                                   |
|          | SVM and kernel methods.                                                                |
|          | **Deadline for HW2.**                                                                  |
| Apr 15   | Dimensionality reduction: PCA and LDA.                                                 |
|          | Using PCA and LDA in combination with other classification and clustering algorithms.  |
|          | **HW3 is posted.**                                                                     |
| Apr 22   | Ensemble methods: Bagging and boosting.                                                |
|          | ADABoost, XGBoost, Gradient Boost.                                                     |
|          | **Deadline for HW3.**                                                                  |
| Apr 29   | Graphs and networks.                                                                   |
|          | How to deal with graph data. Tools and techniques.                                     |
|          | **HW4 is posted.**                                                                     |
|          | **Deadline for Final Project proposals.**                                              |
| May  6   | Perceptron, graph computation and neural networks.                                     |
|          | A tour of different neural network types and architectures.                            |
|          | **Deadline for HW4.**                                                                  |
| May 13   | More on neural networks.                                                               |
|          | Examples and applications.                                                             |
|          | **HW5 is posted.**                                                                     |
| May 20   | Advanced topics, other examples and applications.                                      |
|          | **Deadline for HW5.**                                                                  |
