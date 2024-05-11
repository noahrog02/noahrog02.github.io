---
permalink: /
title: "Recent Applications of Maschine Learning in the Stock Market"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

ignore: [link wird so gemacht](https://google.com)

Introduction
======
Money is a fundamental element of the stock market, where vast sums are exchanged daily. For example, one of the world's largest Exchanges, the NYSE, handles over $2 trillion in transactions each day. The stock market therefore plays a critical role in global finance

**What is the Stock Market**

The stock market is a collection of markets where stocks, are bought and sold. Stocks resemble ownership in their corresponding companies and therefore represent claim on the company's assets and earnings. For companies, the stock market is a crucial mechanism to raise capital and in exchange offering investors opportunities to gain returns on their investment. 

**Evolution of Technology in Stock Market Predictions**

When it comes to predicting prize movements, probably the oldest methods are simple chart analysis and manual calculations. Even though equations might be updated, these methods are still used today. But as the financial world grew, so did the need for faster and more accurate methods. This leads to the introduction of electronic trading systems in the late 20th century, which were soon followed by algorithmic trading. Algorithmic Trading utilized complex mathematical models to make rapid trading decisions. Today, the integration of Big Data and Artifical Intelligence represents a new movement in market prediction. This blog post aims to dive into the Machine Learning aspect of these new Methods, to show what has already been achieved and what might be possible in the future. 

Overview of Machine Learning in Stock Market
======

**Summary of Different ML Applications in Stock Trading**

These methods are used, either standalone or in combination, in various areas of the stock market. The four main areas are:
-Here i will again give a little introduction to each of the areas:-
- Stock price prediction
- Stock movement prediction
- Portfolio management
- Trading strategies
- Subfields that also use ML in some cases (order execution and market making)

**ML and it's relevance in Stock Trading**

Machine learning is a subset of artificial intelligence that involvses algorithms and statistical models that enable computers to perform different tasks without instructions. In this blog post, we will focus on the following ML methods:
-Here i will mention and give a small introduction to each:-
- Recurrent Neural Networks (RNN) are powerful in handling and predicting sequences of data. Unlinke traditional feed forward neural networks that process inputs independently, RNNs have loops within them that allows information to persist. This makes them ideal for tasks where context from previous data helps predicting the next output. So perfect for Stocks which is just time-series data. By variating the standard RNN you can obtain even better results. Two variants are: 
  - Long Short-Term Memory Networks (LSTMs) to avoid the long-term dependency problem. They are practically good in remembering information for longer periods without forgetting important details by having additional input, forget and output gates. These gates determine what information should be remembered or forgotten as data flows through the network. Here, the input gate decides, which data should be used to influence the memory. The forget gate decides which values from the input should be kept or forgotten and the output gate determines what from the current memory state should be output. This makes them extremely effective for tasks involving complex dependencies over time, such as long-term stock behaviour. 
  - Gated Recurrent Units (GRUs) are an evolution of LSTMs trying to simplify the model for faster learning while keeping it's functionality. They achieve this by combining the input and forget gates into a single update gate and add a reset gate to determine how much of the past information should be forgotten. They again only have one output stream which also adds to the improved learning speed of the model.
![RNN-Variants](images/RNN_variants.png)
- GNN
- CNN
- Transformers
- RL
![ML-Variants](images/Different_MLs.png)

Survey of current ML Approaches
======
- Analysis of current literature and findings (related work)

**Dataset and Input**

Input essentially comes down to instrinsic (extracted from the stock itself) and extrinsic (text, fundamental data, industrial knowledge graphs, ...) data. The features of our input are:
- Time series data
- Text in form of news or articles
- Graphs e.g. industrial knowledge graph
- Others like Image/Audio data

**Examples**

Out of every subsection of used ML methods, i will quickly name a few examples based on papers that used these methods.
- Akita et al created a LSTM based model that takes Text and Stock data as input to predict future prices. By taking ten companies and representing them as news, then transforming these news into so called paragraph vectors they manage to provide numerical data for the LSTM that it can work with. $P_{t}$ will become a vector of the numerical news values and $N_{t}$ are the corresponding stock prices at the timestamp t. These two vectors then get concatenated and put into the LSTM as an input. The corresponding output of the network will be $N_{t+1}$ where you can use basic loss functions to train the network. The reason behind taking ten companies and their news at a specific time is to look for a correlation between the impact of news from one company to the stock price of others. This LSTM methods beats the baselines of other RNN methods used in predicting opening stock prices
![Akita_LSTM](images/LSTM_Aktia.png)
- Example 2
- Example 3

Deep dive into a specific ML Method (I am currently thinking about LSTM or RL)
======
- Detailed explanation of the chosen ML method
- Presentation of empirical results obtained from the model
- Advantages and limitations of the method

Case Study and Demos
======
I will provide images/videos here about the chosen ML Method. NeptuneAI or FinRL will be my motivator here. 

Discussion of Weaknesses and Future Directions
======
- critical analysis of the gaps in current methodologies
- insights into potential future developments in ML for stock trading

Conclusion
======
- Summary of findings and final thoughts
- Encouragement for further research and development in this area

References
======
- List of all sources and further reading links
