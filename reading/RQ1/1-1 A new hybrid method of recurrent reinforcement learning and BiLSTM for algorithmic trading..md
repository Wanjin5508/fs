---
tags:
  - "#RQ2"
---

[[A new hybrid method of recurrent reinforcement learning and BiLSTM for algorithmic trading.pdf]]



# Abstract. 
Recently, the algorithmic trading of financial assets is rapidly developing with the rise of deep learning. In particular, deep reinforcement learning, as a combination of deep learning and reinforcement learning, stands out among many approaches in the field of decision-making because of its high performance, strong generalization, and high fitting ability. In this paper, we attempt to propose a hybrid method of recurrent reinforcement learning (RRL) and deep learning to figure out the algorithmic trading problem of determining the optimal trading position in the daily trading activities of the stock market. We adopt deep neural network (DNN), long short-term memory neural network (LSTM), and bidirectional long short-term memory neural network (BiLSTM) to automatically extract higher-level abstract feature information from sequential trading data, respectively, and then generate optimal trading strategies by interacting with the environment in a reinforcement learning framework. In particular, the BiLSTM consisting of two LSTM models with opposite directions is able to make full use of the information from both directions in attempting to capture more effective information. In experiments, the daily data of Dow Jones, S&P500, and NASDAQ (from Jan-01, 2005 to Dec-31, 2020) are applied to verify the performance of the newly proposed DNN-RL, LSTM-RL, and BiLSTM-RL trading systems. Experimental results show that the proposed methods significantly outperform the benchmark methods, such as RRL and Buy and Hold, with higher scalability and better robustness. Especially, BiLSTM-RL performs better than other methods.


介绍了很多可以用于算法交易的深度学习模型, 可以拿来介绍我的研究的AI部分




