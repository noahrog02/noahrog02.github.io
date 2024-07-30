---
permalink: /
title: "Recent Applications of Maschine Learning in the Stock Market"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Seminar University of Stuttgart. By Noah Roggenbuck

In this blog post, I explore the integration of Machine Learning (ML) in stock market prediction and trading. I delve into various ML applications, such as stock price prediction, portfolio management, and trading strategies and present how different ML models have been used to solve these tasks. 

## Table of Contents
- [Introduction](#introduction)
  - [What is the Stock Market](#what-is-the-stock-market)
  - [Evolution of Technology in Stock Market Predictions](#evolution-of-technology-in-stock-market-predictions)
- [Overview of Machine Learning in Stock Market](#overview-of-machine-learning-in-stock-market)
  - [Summary of Different ML Applications in Stock Trading](#summary-of-different-ml-applications-in-stock-trading)
  - [Why use Machine Learning for Stock](#why-use-machine-learning-for-stock)
- [Survey of Current ML Approaches](#survey-of-current-ml-approaches)
  - [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
  - [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
  - [Graph Neural Networks (GNN)](#graph-neural-networks-gnn)
  - [Transformer Based Models](#transformer-based-models)
  - [Reinforcement Learning](#reinforcement-learning)
- [Deep Dive into FinRL](#deep-dive-into-finrl)
  - [Use Cases](#use-cases)
  - [Agents and Their Benefits](#agents-and-their-benefits)
  - [Actor-critic Reinforcement Learning](#actor-critic-reinforcement-learning)
  - [Problem Formulation for Portfolio Management](#problem-formulation-for-portfolio-management)
  - [Training the Agents](#training-the-agents)
  - [Backtesting](#backtesting)
- [Discussion of Weaknesses and Future Directions](#discussion-of-weaknesses-and-future-directions)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Money is a fundamental element of the stock market, where vast sums are exchanged daily. For example, one of the world's largest Exchanges, the NASDAQ, handles over $200 billion in transactions each day [1]. The stock market therefore plays a critical role in global finance.

### What is the Stock Market

The stock market is a collection of markets where stocks, are bought and sold. Stocks resemble ownership in their corresponding companies and therefore represent claim on the company's assets and earnings. For companies, the stock market is a crucial mechanism to raise capital and in exchange offering investors opportunities to gain returns on their investment. 

### Evolution of Technology in Stock Market Predictions

When it comes to predicting prize movements, probably the oldest methods are simple chart analysis and manual calculations. Even though equations might be updated, these methods are still used today. But as the financial world grew, so did the need for faster and more accurate methods. This leads to the introduction of electronic trading systems in the late 20th century, which were soon followed by algorithmic trading. Algorithmic trading utilizes complex mathematical models to make rapid trading decisions. These algorithms still account for 70% of all trades in the stock market [2]. Today, the integration of Big Data and Artifical Intelligence represents a new movement in market prediction. This blog post aims to dive into the Machine Learning aspect of these new methods, to show what has already been achieved and what might be possible in the future. 

## Overview of Machine Learning in Stock Market

### Summary of Different ML Applications in Stock Trading

There are differents fields in which ML is being used in the stock market. The main and most interesting are:
- **Stock price prediction:** This is probably the most common area, where ML will try to predict specific prices of one or more stocks for a specific time interval. ![prediction](images/Prediction.png) Image from [devpost](https://devpost.com/software/stock-portfolio-allocation)
- **Stock movement prediction:** The goal of this approach is the same as direct stock price prediction. But in this case the machine only predicts if the price goes up, down or sideways for a specific time interval. ![MovementPrediction](images/MovementPrediction.png) Image created on [TradingView](https://www.tradingview.com/chart/?symbol=NASDAQ%3AAAPL)
- **Portfolio management:** Portfolio management aims to have a fixed amount of capital allocated to numerous stocks. This basically comes down to a weight distribution problem, where the ML has to decide what stock gets what percentage of the capital provided. ![PortfolioManagement](images/PortfolioAlloc.webp) Image from [medium](https://medium.com/analytics-vidhya/portfolio-optimization-using-reinforcement-learning-1b5eba5db072)
- **Trading strategies:** Providing trading strategies that a human can use to trade in the market is another research area of ML in stock. The goal is to use ML to find strategies like buy or sell signals that a human can easily understand and use.  ![TradingStrategies](images/position-trading.png) Image from [axi](https://www.axi.com/eu/blog/education/trading-strategies)
- **Others:** There are some other use cases where the most common are order execution and market making. 

### Why use Machine Learning for Stock

Machine learning is a subset of artificial intelligence that uses algorithms and statistical models to enable computers to perform tasks without explicit instructions. It is strong at handling large datasets, making it ideal for the stock market where it can analyze extensive historical data. Unlike human-driven technical analysis, machine learning can identify patterns and movements that are not apparent to humans. Additionally, it can integrate various data sources, such as stock prices and news articles, converting them into numerical values to better predict market behavior despite its inherent unpredictability.

![Motivation](images/Motivation.png) Image from [3]

This graph from Liu et al. [3] displays how, with the use of their DDPG agent, they outperformed the Minimum Variance and Dow Jones Index. Three portfolios where one was holding the Dow Jones Index, the second was holding the Minimum Variance Index and the third was managed by an DDPG Reinforcement Learning agent can be seen in this graph. For our purposes it is only necessary to understand that the Minimum Variance and Dow Jones Index move in a way that the market is moving in general as they take many different stocks into account. One is said to beat the market if he manages to gain a higher return rate over a specific time interval than the market is moving in general. For the DDPG Agent of Liu et. al this was strongly the case where they had a near 50% higher return rate than the market in general.

It is therefore very possible to beat markets and make more money than the normal trader by using Machine Learning. We will now come to evaluate different Machine Learning models and how they can be used to fulfill the named applications. 

## Survey of current ML Approaches

Now, to survey different approaches and usecases, we will quickly clear off what kind of input we usually have when working with Machine Learning and the stock market

Input essentially comes down to instrinsic (extracted from the stock itself) and extrinsic (text, fundamental data, industrial knowledge graphs, ...) data. The features of our input are:
- Time series data
- Text in form of news or articles
- Graphs e.g. industrial knowledge graph
- Others like Image/Audio data

Yet every input in some way needs to be converted to a numerical value that the machine can really work with. 

### Used models in Stock Market

In this blog post, we will focus on the following ML methods:

![ML-Variants](images/Different_MLs.png) Image from [4]

### Recurrent Neural Networks (RNN)

are powerful in handling and predicting sequences of data. Unlinke traditional feed forward neural networks that process inputs independently, RNNs have loops within them that allows information to persist. This makes them ideal for tasks where context from previous data helps predicting the next output. By variating the standard RNN one can obtain even better results. Two variants are: 
- Long Short-Term Memory Networks (LSTMs) to avoid the long-term dependency problem. They are practically good in remembering information for longer periods without forgetting important details by having additional input, forget and output gates. These gates determine what information should be remembered or forgotten as data flows through the network. Here, the input gate decides, which data should be used to influence the memory. The forget gate decides which values from the input should be kept or forgotten and the output gate determines what from the current memory state should be output. This makes them extremely effective for tasks involving complex dependencies over time, such as long-term stock behaviour. 
- Gated Recurrent Units (GRUs) are an evolution of LSTMs trying to simplify the model for faster learning while keeping it's functionality. They achieve this by combining the input and forget gates into a single update gate and add a reset gate to determine how much of the past information should be forgotten. They again only have one output stream which also adds to the improved learning speed of the model.

![RNN-Variants](images/RNN_variants.jpg) Image from [nakuri](https://www.naukri.com/code360/library/understanding-an-rnn-cell)

This picture shows the different activation functions, inputs and outputs of the three variants of RNN. 

For example Akita et al [5] created a LSTM based model that takes Text and Stock data as input to predict future prices. By taking ten companies and representing them as news, then transforming these news into so called paragraph vectors they managed to provide numerical data for the LSTM that it can work with. $$P_{t}$$ will become a vector of the numerical news values and $$N_{t}$$ are the corresponding stock prices at the timestamp $$t$$. These two vectors then get concatenated and put into the LSTM as an input. The corresponding output of the network will be $$N_{t+1}$$ where they used basic loss functions to train the network. The reason behind taking ten companies and their news at a specific time is to look for a correlation between the impact of news from one company to the stock price of others. This LSTM method beats the baselines of other RNN methods used in predicting opening stock prices. As there is only a simple LSTM Network used they can adjust parameters by using a simple loss function like MSE. 

![Akita_LSTM](images/LSTM_Aktia.png) Image created in PowerPoint

With this approach they managed to beat the competition in stock price prediction. Using the prediction of a certain price one could use leverage or option trading to amplify even small price movements. 

### Convolutional Neural Networks (CNN)

CNNs are often used for processing and analyzing visual data. They work by automatically detecting and learning hierarchical patterns in images through layers of convolutional filters, pooling, and fully connected layers. This structure allows CNNs to effectively identify and classify objects within images. 

![CNN](images/CNN.jpg) Image from [saturncloud](https://saturncloud.io/blog/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way/)

By slightly changing the problem formulation and constructing an image like input data it is still possible to use these CNN Models in the stock market. To create a grid like input, Lu et al. [6] created a 2D-Array representing a 10-day historical time series where each column is representing a day and the rows represent the opening price, closing price, highest price, lowest price and trading volume of each day. After a simple standarization process this input data is then acting as an input for a CNN Model which will input its extracted features into a LSTM Layer. This LSTM Layer will then produce many different output values that need to be processed by a full connection layer to produce a single closing price $$T+1$$ for the stock. 

![CNN-LSTM](images/CNN_LSTM.jpg) Image from [6]

By using the MSE of the real value and the produced prediction they updated the parameters of the two models.
To evaluate their models performance they compared it's mean absolute error and root mean squared error to different other models in stock price prediction, where their model performed best. Again, using this approach like the simple LSTM approach, one could use the prediction of the machine to trade for single days and amplify the gains by using leverage.

![CNN-LSTM-Results](images/CNN-LSTM-Results.png) Table from [6]

### Graph Neural Networks (GNN)

By using Graph Neural Networks we can create a Machine that is capable of predicting the markets using a graph where nodes may represent companies and edges a correlation between them. In general graphs are used to represent things like social media networks, or molecules. These graphs can be represented by an adjacency matrix. <br>
It is hard to analyse these graphs because of their inconsitent form, where nodes can have different amount of neighbours. This is not the case for image like data that CNNs can work with. In case of an image we have a fixed grid with a fixed size. CNNs fail in working with graphs because of their arbitrary size of the graph, and the complex topology. <br>
GNNs process graph-structured data by iteratively updating node representations through layers. They perform message passing where each node aggregates information from its neighbors. This aggregated information is then transformed using learned weights, followed by applying a non-linear activation function like ReLU. Repeating this across multiple layers, GNNs refine node representations by incorporating progressively larger neighborhoods, resulting in final node features that capture both local and global graph structures, useful for tasks such as node classification and link prediction.

![GNN](images/GNN.png) Image from [saturncloud](https://saturncloud.io/blog/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way/)

Xu et al. [3] proposed a method of using GNNs to predict the behavior of stocks that have already hit a price limit. A price limit in certain markets limits a stock to not rise above a daily limit to prevent excessive volatility. Their model predicts whether a price-limit-hitting stock will close at the price limit (Type 1) or below it (Type 2) by using a Hierarchical GNN. HGNNs combine multiple level of information like stock data, relationships between stocks, and overall market trends. The input to their HGNN includes historical stock sequences, technical indicators and a stock relation graph. The model uses LSTM networks to extract historical features and MLP to extract limit-related features. These features are then processed through hierarchical layers that consider node, relation and graph views using graph convolution and attention mechanisms. The final output is then classified using a fully connected layer to predict wheter a price-limit-hitting stock is Type 1 or 2. 

![HGNN](images/HGNN.png) Image from [7]

By using this information one may short Type 2 price-limit-hitting-stocks and buy Type 1 as they are likely to trade higher the next day. The accuracy using this model can be seen in the table below, where the authors compared the accuracy of different models in two different markets SSE and SZSE. 

![HGNN_Results](images/HGNN_Results.png) Table from [7]

### Transformer based models

Transformer-based models are highly effective in language processing due to their use of self-attention networks, which help retain distant data and understand context by handling entire sequences of data simultaneously. They consist of layers of self-attention and feed-forward networks, enabling them to process and generate text with remarkable accuracy. The two most common transformer based language models are GPT and BERT [9]. By leveraging their capabilities in handling text data, these models have become popular for classifying stock news articles, thereby aiding in predicting market movements. 

![Transformer](images/TransformerBased.png) Image from [researchgate](https://www.researchgate.net/figure/Overview-of-the-proposed-transformer-based-language-model-with-two-transformer-layers_fig1_353838462)

Zhou, Ma and Liu [8] used BERT [9] creating a model that can predict market events. An event in the stock market is for example a so called earings call, which happen quarterly, where the company presents the results of the last quarter and tries to explain future directions. Or maybe a board member of the company buying or selling the stock. All these events are often followed by price movement, where the authors tried to predict if this price movement is positive or negative. <br>
BERT will get a news article of the corresponding company as an input, where every word of the article is called a token. These tokens will then be classified by BERT and transformed in some numerical value $$h_{1}$$ to $$h_{n}$$. Additionally BERT will create a so called $$h_{cls}$$ token that classifies the whole news article in a single numerical value. They then use a low-level and a high level-detector feed forward network that aims to give predictions, wheter a buy or sell event is happening. The low-level detector network will get the classified values $$h_{1}$$ to $$h_{n}$$ as an input and produce a low-level prediction based on these values. This low-level prediction together with the $$h_{cls}$$ of BERT will then be forwarded into the high-level detector which will produce the final high-level prediction. 

![Event_Transformer](images/Event_Transformer.png) Image from [8]

The authors used the predictions to buy or short a stock at the news articles' publish time and close the transaction after two trading days. Their approach grately outperformed the market index. But they warn that it is very important to trade in the same minute that the news article is published, or the machine will have a negative return rate. 

![Event_Transformer_Results](images/Event_Transformer_Results.png) Table from [8]

Win Rate stands for the overall winning rate (rate of transactions that have a return over 0) and big win rate (rate of transactions that have a return over 1%). Ave. Return stands for the average return on each transaction. Exc. Return stands for the total excess returns over the market when starting with $10000 and invest $2000 to each detected trading signal. Num. Trans. stands for the number of transactions (valid trading signals) of each model.

### Reinforcement Learning

Reinforcement learning is probably the most promising model for stock trading. This model involves an agent that takes actions based on a given policy in response to a certain state and reward from the environment. During the learning phase the agent iteratively interacts with the environment, updating its policy parameters based on the rewards received from its actions, using mathematical equations like the Bellman Equation. Ideally, once the learning is complete, the agent will have a table of state-action pairs that have the best rewards for every possible state in the environment.

![RL](images/RL_func.png) Image from [domino](https://domino.ai/blog/transformers-self-attention-to-the-rescue)

Using fixed tables of state-action pairs will not work in an environment like the stock market as there are infinite possible states. Therefore, when training an agent, one has to use a probability distribution for states and their corresponding actions. These can be learned by using variations to the Bellman Equation like Q-Learning and Policy Gradient. We will now look at a detailed example how anyone can use the GitHub repository FinRL (Financial Reinforcement Learning) to create, learn and use a reinforcement learning agent that is capable of trading in the stock market as its environment. 

## Deep dive into FinRL

Financial reinforcement learning (FinRL) [10] is the first open-source framework for financial reinforcement learning. It provides a toolkit for developing and evaluating reinforcement learning algorithms specifically for financial markets. FinRL aims to bridge the gap between finance and machine learning by offering resources and tools for both researchers and practitioners.

### Use Cases

FinRL utilizes many different reinforcement learning agents that do certain trading tasks. These tasks may be Stock Trading, Portfolio Allocation, High Frequency Trading, Cryptocurrency Trading, Market Regulations or User-defined Tasks. The agent will make sequential decisions on FinRL specific environments that are based on market data from [Yahoo! Finance](https://finance.yahoo.com/) for example. 

![FinRL_UseCases](images/FinRL_UseCases.jpg) Image from [GitHub](https://github.com/AI4Finance-Foundation/FinRL)

In this detailed example we want to look at portfolio allocation which, as presented in the motivation part by Liu et al., yielded very good results when using the reinforcement learning agent DDPG. 

### Agents and their benefits

FinRL provides nine different agents of which three are value based. These value based agents can only do single stock trading and are therefore of no use in this example. The other six agents are Actor-critic based and are all suitable for a portfolio allocation problem. These agents only have slight differences as seen in the table:

![FinRL_Agents](images/FinRL_Agents.png) Image from [YouTube](https://www.youtube.com/watch?v=ZSGJjtM-5jA)

### Actor-critic Reinforcement Learning

Actor-critic reinforcement learning is the base of all agents we use and is generally speaking a variation of plain reinforcement learning [11]. In this variation our agent is the so called actor and simply represents a feed-forward network that tries to simulate a policy function. Additionally a Critic-Network tries to implement a Value-Function (in our case a Q-Value). After the Actor-Network decides on a specific action based on the state, the environment will first feed the reward and next state of this action into the Critic-Network, which will compute a Q-Value. A Q-Value is basically a predicted reward for a state-action pair. This Q-Value will be compared to the actual reward to create a so called Advantage Function $$A_{π_{θ}}$$ that is also called TD error with the formular 

$$A_{π_{θ}}(s_t, a_t) = r(s_t, a_t) + Q_{π_{θ}}(s_{t+1}, a_{t+1}) - Q_{π_{θ}}(s_t, a_t)$$ 

The weights of the critic network will now be updated by using this TD error

$$w=w+αA_{π_{θ}}$$ 

The actor network will then also update its weights by using this advantage function. This is done by sampling many different state action pairs and using gradient descent with the gradient: 

$$
\nabla J(\theta) \approx \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t, s_t) A_{\pi_\theta}(s_t, a_t)
$$

Resulting in updating the policy parameters of the actor network with this:

$$
\theta = \theta + \alpha \nabla J(\theta)
$$

![ActorCritic](images/ActorCritic.png) Image from [11]

### Problem formulation for Portfolio Management

To use RL, one has to define what the states, actions and rewards are. In our Portfolio Management approach we want the machine to trade on a fixed number $$D$$ of stocks. For our example we will set $$D=30$$ and simply take the same 30 stocks that are also used by the Dow Jones Index. 

Each **state** $$s$$ in timestep $$t$$ will consist of the vectors $$\vec{p}$$ for the prices of each stock and $$\vec{h}$$ for the amount of shares that we hold of the corresponding stock. The values $$p_{i}$$ are not just simple numbers representing the exact price at time $$t$$ but in our example consist of 16 values that represent high, low, open and close price of $$t$$ but also 12 more indicators that might prove useful to the machine. Additionaly each state has a singe numerical value $$b$$ representing the remaining balance of the portfolio. This number can not go into the negative. 

![States](images/States.png) Image created in PowerPoint

Our **actions** $$a$$ that the actor does are a set of actions on all $$D$$ stocks. The actor will change the values in $$\vec{h}$$ by selling, buying or holding each $$h_{i}$$. 

![Actions](images/Actions.png) Image from [3]

The **reward** $$r$$ of an action $$a$$ simply is the change of portfolio value, where a positive change in value is a positive reward.

### Training the Agents

Each timestep $$t$$ will resemble one trading day, which means the actor will make daily trades. After extracting all the data we need we will simply create an environment that consists of each trading day as a state over the span of 11 years from 2009 to 2020. This is our training set. We will also create a set of states for 2020 to 2022 as our backtesting set. For FinRL, one has to additionally define the cost of opening or closing a position, which basically simulates a brokers fee. FinRL itself does not come with implementations of RL algorithms but provides a framework for applying existing RL algorithms to financial trading environments. Thats why, in our case, we will use the Actor-Critic RL Agents from [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/). <br>
This means that the only part for us to do as a developer that uses FinRL, is to set up the environment and configure parameters like the learning rate, number of steps until policy update and entropy coefficient to encourage exploration of the SB3 Agents. In our case we left the parameters of every agent at default. <br>
In the actual training phase the agent will learn based on 50000 time steps. This means that the agent will go over the whole training set approximately 16 times as there are 3000 trading days in our training set. The number of steps is set to five, which means that the agent will update its policy after every fifth timestep. 

### Backtesting

Once every agent has been trained, backtesting will be the last step to see how the agent would perform in the real market. First we will set up a sample portfolio that is fully invested in the Dow Jones Index and another that is fully invested into the Mean Variance Index. These two portfolios will create the baseline to see wheter the agents manage to beat the market or not. Lastly we will let the agents work on the backtesting environment. By tracking the value of each portfolio each day we get the following graph: 

![FinRL_Results](images/FinRL_Results.png) Image from [10]

The first thing to see is that every portfolio moved in the same pattern, basically having the same up and down movements. Yet no agent manages to fully beat the market to make more profit than the Mean Variance Index. Not even the DDPG agent or the TD3 (improvement to DDPG) agent outperformed the market. This is contradictory to the graph by Liu et al. presented in the motivation part where they managed to greatly outperform the market. The exact reason for the clear difference might come from fine tuning aspects but also from the fact that the DDPG agent from Liu et al. never stopped training. Even when trading in the real market they enabled the agent to continue learning so that it catches new trends. <br>
In the given example usage of FinRL we did not consider the fact of continuous learning as this requires a decent amount of extra work. 

## Discussion of Weaknesses and Future Directions

Despite the promising applications of machine learning in stock market predictions, several weaknesses persist. Future directions for research and development are crucial to address these limitations and enhance the effectiveness and reliability of these models. 

| **Weakness** | **Future Direction** |
|--------------|----------------------|
|Simple ML like the basic example of FinRL does not beat the baselines of the market|Try to make even complicated ML approaches more accessible and fine tune the basic models|
|The stock market still remains unpredictable. There are times where the whole market simply breaks and a machine will loose money if it does not know that these times are coming|It is possible to incorporate even more data sources. A machine could be developed that gets input from stock data, news data and maybe even other sources all at once. It will see devastating events coming|
|There is still no real time model that trades live on the market. For example the event prediction transformer based model will only work properly if there is zero latency between the publish time of a news article and the immediate buy or short action of the underlying stock|Develop such a real-time prediction model with minimal latency|
|Models are not robust among differnet stocks or markets. The models will always only work on the conditions one has trained them on|Develop a model suitable for all markets. Like a professional human trader|

## Conclusion

In this post we have only looked at a few different examples of how ML can be used to predict the stock market. There are hundreds of other papers presenting their own approach with all of them having their own benefits and disadvantages. While machine learning offers significant potential for stock market predictions, it is not without its challenges. We did not discuss how large market players like hedge funds use ML, but some of them might have cracked the code to insanely beat baselines by using fine tuned ML approaches that were mentioned in this blog post. <br>
Future work might prove promising results and it is safe to say that ML will play a big role in the stock market heading in the future. But it remains a question how healthy that might be for the market. 

## References

- [1] Nasdaq Trader. "Daily Market Summary." *Nasdaq Trader*, Nasdaq, Accessed 30 July 2024, https://www.nasdaqtrader.com/Trader.aspx?id=DailyMarketSummary.
- [2] Quantified Strategies. "What Percentage of Trading Is Algorithmic?" *Quantified Strategies*, Accessed 30 July 2024, https://www.quantifiedstrategies.com/what-percentage-of-trading-is-algorithmic/.
- [3] Liu, Xiao-Yang, et al. "Practical deep reinforcement learning approach for stock trading." arXiv preprint arXiv:1811.07522 (2018).
- [4] Zou, Jinan, et al. "Stock market prediction via deep learning techniques: A survey." arXiv preprint arXiv:2212.12717 (2022).
- [5] Akita, Ryo, et al. "Deep learning for stock prediction using numerical and textual information." 2016 IEEE/ACIS 15th International Conference on Computer and Information Science (ICIS). IEEE, 2016.
- [6] Lu, Wenjie, et al. "A CNN‐LSTM‐based model to forecast stock prices." Complexity 2020.1 (2020): 6622927.
- [7] Xu, Cong, et al. "HGNN: Hierarchical graph neural network for predicting the classification of price-limit-hitting stocks." Information Sciences 607 (2022): 783-798.
- [8] Zhou, Zhihan, Liqian Ma, and Han Liu. "Trade the event: Corporate events detection for news-based event-driven trading." arXiv preprint arXiv:2105.12825 (2021).
- [9] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding.", NAACL-HLT 2019.
- [10] AI4Finance Foundation. "FinRL: Financial Reinforcement Learning." *GitHub*, 2024, https://github.com/AI4Finance-Foundation/FinRL.
- [11] Lapan, Maxim. "The Actor-Critic Reinforcement Learning Algorithm." *Medium*, 6 Aug. 2018, https://medium.com/intro-to-artificial-intelligence/the-actor-critic-reinforcement-learning-algorithm-c8095a655c14. Accessed 30 July 2024.

