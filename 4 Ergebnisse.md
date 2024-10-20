

- Die zur Beantwortung der Forschungsfragen verwendeten Literatur und theoretischen Grundlagen.
- Ich hoffe, aus den Referenzen die Verbindungen zwischen verschiedenen Forschungsfragen zu identifizieren und diese Fragen in einer spezifischen logischen Reihenfolge zu arrangieren.
- Beachtung der Kombination verschiedener Autorenmeinungen.

- [ ] 思考如何组织这部分的具体内容, 
	- [ ] 1. 我想按照不同的投资领域, 例如股票, 大宗商品, 外汇, 加密货币, 投资组合等划分小标题
	- [ ] 2. 也可以按照不同的技术路线, 例如~~统计, 算法,~~ 机器学习, 深度学习, 深度强化学习

?? 我更倾向于采用第二种方案, 因为我的研究等题目是关于AI技术的, 所以在这一章也采用技术路线作为标题是合理的。标题结构如下:

## 4.0 常规算法交易方法
根据算法交易的发展过程, 首先被应用于实践的方法是基于统计学和数学的方法。随着AI技术和相应硬件的发展, 基于数学和统计学的算法交易方法不断被新的AI技术补充。因此本研究在介绍相关的AI工具前首先阐述常规的算法交易方法是有必要的。

1-7: 
由于算法交易的频率极高的特性, we cannot observe directly whether a particular order reflects AT or not. 因此, we use the sum of order submissions in a minute as a proxy for AT

### 4.1.1 an automated exit and re-entry strategy

根据算法交易的发展过程, 首先被应用于实践的方法是基于统计学和数学的方法。随着AI技术和相应硬件的发展, 基于数学和统计学的算法交易方法不断被新的AI技术补充。因此本研究在介绍相关的AI工具前首先阐述常规的算法交易方法是有必要的。

s-1-2
在常规的算法交易研究和实践中, 常见的一个方法是Price jump methods, 该方法可以被应用于financial risk management and market volatility analysis。Price jump在算法交易领域指的是over short time frames内突发的价格变动, 这种价格波动反映了市场的波动 (s-1-2)。

(s-1-2)等人提出了一个novel automated exit and re-entry strategy with the primary aim of anticipating price jumps in a timely manner and then analysing market volatility until a suitable point of market re-entry has been identified. Intraday traders utilise small trades and intraday price fluctuations to accumulate portfolio returns without keeping a position open overnight. Since high market liquidity is a requirement for intraday trading, the Foreign exchange (Forex) market is often the asset of choice (s-1-2).

s-1-2
VaR analyses are a form of risk management in which it is assumed that the assets’ values depend purely on their prices and are not influenced by the relevant trade volumes or trade types (buys or sells) (Gourieroux & Jasiak, 2010). 借助VaR分析, 投资组合在接下来M个交易日中的最大期望损失能够被计算。本策略涉及的另一个方法是Entropy, entropy is an indication of system disorder or chaos。In the context of finance theory, entropy is often associated with probability theory and is a measure of system uncertainty. Broadly speaking, entropy is applied to portfolio selection (a subfield of risk management) and asset selection

the automated exit and re-entry strategy comprises two underlying phases. During Phase 1, a VaR analysis is performed to identify price jumps. Thereafter, Phase 2 involves carrying out an entropy analysis to identify an appropriate point for market re-entry (s-1-2). 在两个案例研究中, (s-1-2)等人通过验证得到了积极的结果, 并给金融市场上的价格波动带来的潜在损失赋予了一个货币价值。

![[Pasted image 20241012155933.png]]
A process diagram of the proposed automated price jump exit and re-entry strategy.



## 4.1 机器学习
特点和优势

每个子章节用于概括文献中使用机器学习技术的文献所涉及的投资市场

值得注意的是, 机器学习通常被用于预测某种资产未来的价格走势, 但是无法直接输出潜在的投资决策。

*分析检索到的文献中 , 哪些研究采用了机器学习.*: 

1-3: 介绍了多种ML算法
~~s11: 分类算法: 朴素贝叶斯~~
s-1-18



在检索到的涉及AI和算法交易的文献中, 有些学者采用常规的机器学习的方法进行算法交易. machine learning algorithms have proven to be instrumental tools for devising trading strategies that can optimize market inefficiencies (1-3). 出于对算法交易的不同的理解, 有的研究者认为AI技术对算法交易的影响体现在, 借助机器学习算法, 金融市场上的相关指标可以被获取(s-1-1)。也有些研究者试图通过不同的算法来预测金融市场的趋势，但随着市场复杂性的增加，市场预测的难度也在增加(s-1-2)。但是也有学者持相反观点, 认为仅仅使用预测性分析来预测市场的趋势或者某只股票的未来涨跌是远远不够的, 还要采用更智能的方式从而得到更好的投资策略( )。

为了实现准确的预测, 以机器学习为代表的非线性预测模型被研究和应用。这些常见的预测模型通常被分为两类, 分别是分类器(例如支持向量机, SVM)和回归器(例如 Support Vector Regression (SVR)) (s-1-18), 分别用于预测离散数据和连续数据。 (1-3)等人借助python的scikit learn library提供的Ridge Regression, Ada-Boost, Light-GBM, XG-Boost, Linear Regression, and Cat-Boost 等机器学习算法, 专门针对美国股票市场制定了算法交易策略。The objectives of algorithmic trading are to utilize advanced data analytics and computing power to pinpoint beneficial trading opportunities and trends that may be too complicated or rapid for human traders to acknowledge and act upon (1-3).

### 4.1.1 分类算法用于市场趋势预测
通过分类算法对某个特定的金融市场的未来趋势进行预测, 能够给交易者带来收益的机会。实际上, market directional也是算法交易的一种方式(2-1)。方向性交易指的是, 投资者根据其对市场价格的走势的预期进行交易。如果交易者推测未来的价格会上涨, 那么他会买入资产并等待资产价格上涨后通过卖出该资产获利, 这个过程也叫做多。当交易者预期到资产价格下降, 那么他会选择借入该资产并将该资产在市场上卖出, 等资产价格下降后, 交易者会再次将等量的该资产买入并归还, 这个过程也叫做空。

(s-1-18)等人对比了来自不同国家的10个股票市场上, Support Vector Machines (SVM), Random Forest (RF) and Naive-Bayes (NB)作为预测算法, 分别对current daily closing price directions和the next day closing price direction的预测效果进行比较。这些预测算法的独立变量是技术分析指标, 包括Simple Moving Average (SMA), Weighted Moving Average (WMA), Exponential Moving Average (EMA), Momentum, William R%, Moving Average Convergence Divergence (MACD), Relative Strength Indica- tor (RSI), Accumulation/Distribution Oscillator (ADO) and Commodity Channel Index (CCI)。这些技术指标均通过每个交易日的收盘价计算。同时, 模型的非独立变量则是收盘价的趋势, 即either upwards or downwards (s-1-18)。

但是通过(s-1-18)等人的验证, 上述以多种技术指标为自变量的几个机器学习模型are virtually useless in providing useful information for trading decisions regarding next day direction。在他们的研究中, 这些分类模型在金融市场走势预测方面, 并没有展示出相较于随机模型的优势。






### 4.1.2 回归算法用于市场趋势预测
(2-1)等人采用了基于回归算法的机器学习模型, RF和SVM, 通过量化指标对比特币期货交易市场上的价格波动趋势进行了预测。The significant volatility and high 24/7 trading volume give a good opportunity for algorithmic high frequency trading (HFT)(2-1), 尤其是相较于股票市场. 由于波动性无法被直接从期货价格的收益中观测, (2-1)等人借助一个代理来获取historical realized volatility, 即variance at time t of returns—the square of returns.  

在随机森林模型中, 每个决策树的输出的均值决定了随机森林的最终输出。When constructing each decision tree, a bootstrap sample is generated by randomly selecting observations from training dataset with replacements. Using the randomly chosen number of features leads to the correlation between the trees to be low, thereby reducing overfitting. 为了预测the square of conditional volatility, 收益的滞后平方(lagged square of returns)被作为模型的输入使用。类似地, 在SVM模型中, 滞后值同样被用于预测squared conditional volatility. 但是SVM divides the input space using hyperplanes using kernel functions (2-1).

### 4.2 聚类算法



## 4.2 深度学习

1-4, 
1-5: CNN-LSTM


Recently, the algorithmic trading of financial assets is rapidly developing with the rise of deep learning. As part of a broader family of machine learning methods, deep learning is often used to forecast stock prices or trend movements in order to build financial trading strategies (1-1).

相较于机器学习方法, 采用深度学习方法从sequential trading data中提取高层次抽象信息(1-1)的研究贡献的数量更多. 研究者尤其热衷于采用深度学习模型LSTM以及相关的多种变体, 例如BiLSTM进行特征提取, 从而训练一个深度学习模型(例如MLP, CNN, 和LSTM)来预测资本市场的趋势。

还有一种和生成式AI强相关的方法: Transformer, 但是直接相关的文献只有一篇




### MLP
(1-1)等人将多层感知机(又称为深度神经网络, DNN)和强化学习方法结合, 将神经网络的20个输入特征经过一个全连接层以及激活函数tanh转化成一个[-1, 1]之间的输出。但是, 从时间序列数据中提取信息本身就是一个复杂的任务. 同时, 由于MLP性能的局限性, 无法准确从序列数据中提取完整的状态(1-1)。因此, 结构相对简单的MLP不是算法交易的首选。

### CNN

### LSTM & BiLSTM
为了更高效地从金融市场的时间序列数据中获取长期和短期的信息, 结构更加复杂的LSTM被引入。the BiLSTM consisting of two LSTM models with opposite directions is able to make full use of the information from both directions in attempting to capture more effective information. The LSTM/BiLSTM is used to approximate the decision function, which maps the state space to the action space, and then maximizes the Sharpe ratio by the gradient ascent method to generate the best trad- ing strategy, respectively (1-1). 如图2提到的LSTM的结构可知, 神经网络的隐藏层的状态不仅仅和当前时刻的输入有关, 也和上一个时间点的隐藏状态有关。







## 4.3 深度强化学习
- 本研究的重点, 也是当前算法交易领域最热门的主题
- 我检索到的文献中, 采用强化学习和深度强化学习的文章数量最多

前面提到的深度学习方法的好坏主要依赖于预测的准确性, 同时深度学习算法存在内在的难以避免的过拟合问题(1-1). 此外, 深度学习方法无法处理算法交易所要求的连续且快速的决策制定, 因此机器学习和深度学习方法通常需要和强化学习方法被关联应用。Compared with forecast-based methods, *deep reinforcement learning* enables mapping from state space to action space through continuous and online self- learning (1-1).

强化学习被更加广泛地应用于多种金融市场上的算法交易, 例如外汇市场, 股票市场, 期货市场等。在目前的文献检索结果中, 强化学习是被最广泛研究的一种方法。但是基于强化学习模型的科研与实践的复杂性是不可忽视的. 

deep reinforcement learning has achieved remarkable success in solving complex sequential decision problems, and therefore more and more research is focusing on the combination of deep reinforcement learning and investment decision-making. It can extract features directly from high-dimensional raw financial data in a ***deep learning*** module, and then find optimal *dynamic* trading strategies to maximize risk-adjusted returns by interacting with the environment in a ***reinforcement learning module*** (1-1).


### RRL
深度强化学习的一个基线模型是 the recurrent reinforcement learning (RRL) algorithm with return as the input and the difference Sharpe ratio as the objective function for single assets and portfolios with a transaction cost. the RRL method only uses a simple full connection and a hidden layer network to jointly generate the trading signal through the previous trading signal and the return sequence  (1-1)
但是这一个基线模型extracts features from time series data in a linear manner, but financial markets are noisy, and a nonlinear model is needed to extract higher-level features. In contrast, deep learning combines features through multilayer network structures and nonlinear transformations with strong perception and representation capabilities (1-1). 因此这个基线模型在文献研究的结果中被用作对照组, 从而检验不同研究者提出的新的更复杂的模型的性能。

![[Pasted image 20241016171525.png]]
Fig. 1. Trading system based on recurrent reinforcement learning.

In this strategy, an agent can trade a fixed position size in a single security. It assumes that the trader can take a short, neutral, or long position of constant magnitude with the objective function of maximizing the difference Sharpe ratio.(1-1). 





### Hybride Methode
1-1
~~hybrid method of recurrent reinforcement learning and BiLSTM for algorithmic trading~~
deep reinforcement learning, as a combination of deep learning and reinforcement learning, stands out among many approaches in the field of decision-making because of its high performance, strong generalization, and high fitting ability.
作者在提出的一个新颖的混合方法, 尝试深度学习和强化学习的结合从而figure out the algorithmic trading problem of determining the optimal trading position in the daily trading activities of the stock market.
作者等人 adopt deep neural network (DNN), long short-term memory neural network (LSTM), and bidirectional long short-term memory neural network (BiLSTM) to automatically extract higher-level abstract feature information from sequential trading data, respectively, and then generate optimal trading strategies by interacting with the environment in a reinforcement learning framework (1-1). 

BiLSTM-based approach can outperform other methods in the U.S. index market 的原因在于, BiLSTM能够充分利用两个方向从而更高效地从数据中获取信息(1-1)。


![[Pasted image 20241016173902.png]]
Fig. 2. The structure of DNN/LSTM/BiLSTM-RL.

Firstly, DNN, LSTM, and BiLSTM were applied respectively to replace decision functions to map the state space to action space, and then maximize the Sharpe ratio by gradient ascent method to generate optimal trading strategies, respectively (1-1).

Traders learn strategies through trial and error exploration, take action and receive positive or negative reinforcement depending on the results. A trading performance function U(θ), such as profit, utility, or risk-adjusted return, is used to directly optimize the trading system parameters θ. Therefore, reinforcement learning was adopt to update the weights in a deep neural network (such as DNN, LSTM, and BiLSTM) via gradient ascent in the utility function U(θ) (1-1). Figure 2 shows the structure of DNN/LSTM/BiLSTM-RL .


用于构建一个基于深度强化学习的算法交易系统的几个基础元素如下:
- θ denotes the deep learning model parameters,
- State: All market information(e.g. Price at time t is pt) for underlying financial assets forms the state of the environment. Here, the returns of the past M trading days are integrated as the agent’s inputs at t time, defined as xt =[rt,...,rt−M].
- Action: The agent in the trading system will try to maximize the Sharpe ratio in the given time series (State). Ft is calculated by the output of the trading system and represents the trading decision at time t.
	- 如果Ft的值为 -1, 那么卖空操作(short sell)被执行
	- 如果Ft的值为0, 那么交易系统不执行任何操作
	- 如果Ft的值为1, 那么交易系统执行做多操作(long buy)
- Reward Function: a trading system‘s return Rt is realized at the end of the time interval (t, t + 1], including the profit or loss during (t, t + 1] and the transaction cost. 
- Utility Function: A trading system can be optimized by maximizing a performance function, such as a utility function of profit, wealth, or a risk-adjusted return like the Sharpe ratio.


### incorporation of sentiment analysis into RL
price-based features can be viewed as the minimum information required for modelling the state, specifically incorporating the latest price history alongside a set of technical indicators to extract some insight into the most likely evolution of the stock price in the future (2-3). 为了弥补仅仅依靠技术指标的算法交易策略的缺点, (2-3)等人通过对特斯拉公司的财报的sentiment analysis对agent所处的环境进行增强. 他们通过模拟股票市场对被sentiment analysis强化的深度强化学习模型进行评估, 并观测到in the augmented environment, the agent was able to increase its cumulative reward in the testing period by up to 70% (2-3).

为了构建一个增强的强化学习环境, (2-3)等人给常规的仅仅包含量化指标的状态空间增添了新的特征, 这些特征是被基于余弦相似度的sentiment analysis所计算出的, 包含如下情态类别: Negative, Positive, Uncertainty, Litigious, Constraining, Interesting. To ensure that all features may contribute to the model’s performance, all features in the window were normalised to a common scale. 

The state array is then reshaped to fit the input requirements of our LSTM network, which takes the shape of a three-dimensional array defined by the following dimensions: (batch_size, time_steps and features) . batch_size 和 time_steps是神经网络中可以被调节的超参数 (2-3). 用于训练的模型的结构如下表X所示.

![[Pasted image 20241020160048.png]]

### Directional Change RL (DCRL)






