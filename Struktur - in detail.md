![[Pasted image 20240813125052.png]]


# 0 Verzeichnis
- [[#1 Einleitung]]
  - [[#1.1 Motivation]]
  - [[#1.2 Zentrale Begriffe]]
    - [[#1.2.1 Finanzmarkt]]
    - [[#1.2.2 AI Technology]]
  - [[#1.3 Forschungsbeitrag]]
- [[#2 Theoretischer Hintergrund]]
 - [[#2.1 Hochfrequenzhandel (HFT)]]
 - [[#2.2 Maschinelles Lernen]]
 - [[#2.3 Deep Learning]]
 - [[#2.4 Reinforcement Learning]]
 - [[#2.5 Heuristische Algorithmen]]
- [[#3 Methodik]]
- [[#4 Ergebnisse]]
  - [[#4.1 Maschinenlernen]]
    - [[#4.1.1 Aktienmarkt]]
    - [[#4.1.2 Kryptowährungen]]
  - [[#4.2 Deep Learning]]
  - [[#4.3 Reinforcement Learning]]
    - [[#4.3.1 Aktien]]
    - [[#4.3.2 Futures]]
    - [[#4.3.3 Kryptowährungen]]
    - [[#4.3.4 Forex]]
  - [[#4.4 Optimierungsalgorithmen]]
    - [[#4.4.1 Heuristische Algorithmen]]
    - [[#4.4.2 Manuelle Anpassungen]]
- [[#5 Diskussion]]
  - [[#5.1 Methoden und Praktiken in der Literatur]]
  - [[#5.2 Chancen und Herausforderungen durch AI]]
  - [[#5.3 Reflexion und Ausblick]]

既然是突出AI技术对AT的影响, 那么我希望按照不同的AI技术划分第四章 Ergebnisse 的小标题, 目前计划:
- 统计方法
- 机器学习
- 深度学习
- 强化学习
- (深度强化学习)




# 1 Einleitung
## 1.1 Motivation

近几十年来, 算法交易正逐渐成为金融交易的趋势(1-1). 这是因为大量的时间和资源被投入到算法交易的设计和实现中, 从而帮助投资者在金融市场获取有竞争力的信息优势(s-1-2)。但是金融市场本身具有的复杂, 不确定, 以及动态的特性(s-1-4), 使得金融交易面临挑战, 例如经济指标、投资者情绪以及其他市场参与者的行为等因素都会影响金融交易(s-1-15)。为了获取更高的交易利润, 新的技术和算法被不断地应用到算法交易策略中。相比于传统的交易方法, 算法交易在速度、精度、理性程度、处理能力以及警惕性方面占据优势(s-1-4)。


自从以chatgpt为代表的生成式AI兴起以来, 不同专业领域的研究人员对于AI在各自领域的应用以及影响产生了浓厚的兴趣。AI也正在改变人们日常生活的方方面面, 并把人类从一些复杂的工作中解放出来。金融领域由于其复杂的特性, 从多年以前就是AI研究的热门领域。许多金融机构, 例如银行、基金管理公司和资产管理者都给机器学习、深度学习、强化学习等AI技术赋予了较高的优先级, 并借此优化其投资策略(2)。
算法交易从早期仅仅依赖于统计学模型和计量经济学模型(g-1, 1-2)以及高频交易(s-1-10)获取单一资产投资收益最大化, 发展到现在基于深度强化学习技术的投资决策辅助工具(s-1-4), AI技术的发展不能被忽视。 在现代化的算法交易中, 包括机器学习, 深度学习等AI技术已经被广泛地应用于数据处理和预测, 从而获取交易信号(1-1)。这是因为机器学习等技术又能力弥补传统方法在发现隐藏关系和pattern中的不足(1-3)。
尽管已经有大量学者通过实际案例, 对于AI技术在算法交易中的应用展开了深入的研究, 
有的研究人员倾向于通过对照实验, 分析不同的AI技术在提高投资收益以及降低交易风险方面的性能差异, 例如 (1-1) 对比了 recurrent reinforcement learning 和 BiLSTM组成的混合方法和单一方法在美国股票市场上的表现. 还有些研究人员针对某个特定的金融市场, 提出一个基于AI的专门的算法交易模型, 例如, (1-5)等人提出了一种遗传算法和AI结合的加密货币交易策略。


尽管通过不断迭代的AI技术获取高额的投资收益在算法交易领域的研究中扮演了重要角色, 但是技术变革带来的影响同样具有研究意义. 得益于大规模可获取的高质量金融数据和AI领域的突破性进展, 新的更复杂的算法交易策略也得以被建立(1-3)。随着近几十年来AI技术的不断发展, 算法交易几乎涉及所有的金融市场, 例如股票市场, 期货市场, 虚拟货币交易所等等。但是目前学术界尚未从技术的角度对算法交易中AI引发的影响进行总结性分析, 这是因为算法交易作为一个复杂的交叉学科, 同时涵盖了经济学, 金融学和计算机科学等多个学科门类。同时, 算法交易的复杂性也增加了不同学术背景的研究者进入这一领域的门槛。因此本研究希望能够通过系统性文献分析的方法, 深入探讨当前学术界针对算法交易的方法和实践的最新研究成果。并在此基础上, 进一步研究AI技术的发展对算法交易行业带来的机遇和挑战。






## 1.2 Zentrale Begriffe
在这一节, 本研究相关的常用概念被简单阐释和区分, 从而让来自不同研究领域的读者们对于算法交易涉及的不同专业术语的含义具备初步的理解。同时, 这些概念在本研究中所代表的具体含义也会被限定, 从而避免读者产生误解。这些概念按照其所属的学科领域分成两类, 即金融市场相关的概念和AI技术相关的概念。

(这部分可以随着研究的深入不断更新)
(思考是否需要制作一个Venn图, 将两个领域关联起来)

### 1.2.1 Finanzmarkt
这里介绍常见的金融市场相关的概念, 以及市场内部交易的对象。金融领域常见的技术指标会在第二章理论基础中介绍。 
包括:
- 算法交易的定义(与量化交易的关系和区别, 引用), 发展历程, FinTech
- 常见的金融交易市场, 例如股票, 外汇, 期货, 加密货币......
- 描述金融交易的专业术语, 上面提到的投资标的物的基本原理
- 用于衡量不同资产的投资回报率的指标, 例如最常见的是夏普比率
- 

- [ ] 注意这里不需要分小节, 只需要按照从大到小的逻辑顺序, 即从整个金融市场, 到细分市场类型, 再到术语, 再到技术指标。
#### 算法交易
在量化金融领域存在几个相似的概念, 例如量化交易, 算法交易, 高频交易和自动化交易。 为了避免读者混淆这些概念, 它们的具体定义会被详细阐释。尽管有些学者倾向于将这些概念严格区分, 但是这些概念实际上是投资领域不同的AI应用。Quantitative trading uses computer algorithms and programs based on basic or complicated mathematical models to discover and profit from trading opportunities. Algorithmic trading executes orders based on time, price, and volume using pre-programmed trading instructions. High-frequency trading, which is commonly abbreviated as HFT, is a type of trading that makes use of highly advanced computer algorithms to execute a huge number of orders in a very short amount of time (fractions of a second). An automated trading system (ATS), a form of algorithmic trading, employs computer software to issue buy and sell orders and automatically send them to a market center or exchange (2). 

尽管这些概念的准确定义存在明确的界限, 但是他们彼此之间关联也不能被忽视。例如, 高频交易(HFT)的研究和实践同样也可以被AI技术实现(s-1-10)。事实上, 根据文献检索的结果, 大多是近5年内的研究并没有对算法交易和高频交易的概念进行明确区分。因此, 考虑到本研究的重点仅局限在AI对算法交易的影响, 本研究采用(1-7)中的定义, 即高频交易在本研究中被视为算法交易的一个子集。

有些学者认为, 算法交易也被认为是black-box trading or automated trading并借助计算机上部署的AI模型或者预先定义好的量化交易规则辅助交易者进行投资决策(1-3)。也就是说, 算法交易即可以被通过基于预先设定的规则实现, 也可以通过机器学习的方法实现(s-1-4, s-1-5)。传统意义上的算法交易是基于人类专家的经验或者预设的规则的, 例如跟踪趋势或者mean reversion strategies(1-1)。而在基于机器学习的现代算法交易中, 计算机首先在历史数据上被训练然后在没有人类干涉的情况下进行交易(s-1-4)。这个交易过程可以被看作是一个以收益最大化和风险最小化为目标的决策过程(1-1)。正是因为AI的决策过程不会像人类那样受到情绪波动的影响, 因此计算机的投资决策不会受到情绪带来的负面影响(s-1-4)。



#### Sharpe ratio
夏普比率是一个评估一次交易的performance的指标, 该指标能够衡量交易策略的有效性(s-1-3)。夏普比率的计算可以通过以下公式实现 :
![[Pasted image 20240920125122.png]]
其中, Rp和Rf分别表示当前交易中的投资组合的回报和无风险回报, 分母则是投资组合回报的标准差.

无风险收益

#### 用于时间序列分析的蜡烛图(candlestik)
待定
![[Pasted image 20240920234216.png]]




### 1.2.2 AI Technology
- 目前狭义上的AI指的是生成式AI, 或者AGI, 而广义上的AI包括了机器学习, 深度学习和强化学习算法在训练数据集上训练好的模型。在算法交易领域, 目前尚未有生成式AI大规模的应用, 因此我选择在本文中采用广义的AI定义。事实上大多数学者的研究也是这么做的 
- 机器学习的定义
- 深度学习的定义, 与机器学习的区别
- 强化学习的定义, 与深度学习的区别。
- 三种技术分别适合解决的问题类型不同, 因此不存在最好的模型, 研究者需要根据实际项目选择最合适的

- [ ] 同样不需要再分小节, 用分段的方式, 按照从大到小的逻辑顺序, 即 AI 的定义以及进一步的各种学习的定义。

为了给读者提供一个关于本研究的基本的理解, 针对有关于AI的概念进行区分是有必要的。AI comprises any technique that enables computers to mimic human behavior and reproduce or excel over human decision-making to solve complex tasks inde- pendently or with minimal human intervention (1 Russell and Norvig 2021).在下图中, AI相关的概念之间的关系被展示。其中, AI囊括了机器学习, 同时深度学习和强化学习是机器学习的子类. 
机器学习作为一个含义宽泛的概念, 很多研究者在论文中将机器学习等同于AI, 甚至使用机器学习这个上位概念指代更加确切的深度学习和强化学习。本研究采用明确的定义, 即机器学习是与其他AI技术并列的概念. 当机器学习作为上位概念泛指其他AI相关技术时, 本研究会将其替换为AI。
机器学习算法能够为算法交易提供一个很有效的工具, 这是因为它允许同时分析一大批金融市场的指标(s-1-1)。 这个分析过程是通过机器学习算法提取金融市场的数据patterns来实现的(2)。根据所解决的问题和可用的数据, 机器学习可以被分为两种类型, 有监督学习和无监督学习(1)。在算法交易中, 有监督学习可以被用于预测某个资产未来的价格走势。但是, According to the efficient-market theory, stock prices should reflect all of the information that is currently accessible, and any price changes that are not based on newly revealed information are therefore intrinsically unpredictable (2).

![[Pasted image 20240921004149.png]](2)

DL is particularly useful in domains with large and high dimensional data, which is why deep neural networks outperform shallow ML algorithms for most applications in which text, image, video, speech, and audio data needs to be processed (1 LeCun et al. 2015). 深度学习通常使用深度神经网络, 该网络典型地包含了超过一个隐藏层, 并且人工神经元被在网状的深度神经网络结构组织。

Reinforcement learning is much more focused on goal-directed learning from interaction than are other approaches to machine learn- ing. Reinforcement learning is concerned with the sequential interac- tion of an agent with its environment. First of all, the agent observes the environment of state, then the agent executes the action resulting from its policy and receives a reward as a result of its action (s-1-8)

Deep Reinforcement Learning (DRL) combines the perceptual capability of deep learning and the control decision making capability of reinforcement learning to learn the mapping between financial market states and trading decisions by interacting with the environment (s-1-8).


## 1.3 Forschungsbeitrag
> Der erwartete Beitrag dieser Studie besteht darin, durch eine systematische Literaturübersicht die oben genannten (zwei oder drei) potenziellen Forschungsfragen zu beantworten. Dies soll anderen Forschern und Entwicklern eine *fundierte theoretische Grundlage* bieten.
--> 导师说, 理论基础在FS中仅仅扮演一个次要的角色, 因为 FS 的目的是从 RQ 中获取 ***Erkenntnisgewinnung***. 即einen aktuellen Stand der Forschung zu dem Thema erarbeiten



算法交易作为一个跨学科且对数学和计算机算法有着较高要求的学科, 给来自不同行业背景的研究者和初学者带来了较高的进入门槛。例如, 金融学领域的研究人员在缺乏对AI技术对了解的情况下, 很难无障碍地阅读算法交易和相应的人工智能技术的的论文。而一部分开发者和信息技术从业人员在不了解金融市场规则和投资策略的情况下, 开发出高性能的算法交易工具也是很困难的。同时, 来自不同领域的研究人员在进行学术讨论时, 也会经常面临双向沟通效率低的难题。
在这个系统性文献研究中, 期望实现的贡献是通过对学术界关于算法交易的最新研究成果以及最佳实践的分析, 为其他研究者提供关于该领域中常用的解决方案和实现这些方案所需要的知识基础。本研究还致力于通过对其他研究者提出的理论和进行的实践的总结, 讨论AI的引入所带来的机遇和挑战. 本研究对于金融交易的参与者们以及其他利益相关者有不同的意义。
因此, 本研究希望在AI产品被大规模应用的背景下, 为来自不同领域的研究者建立一个稳固的知识基础, 同时在不同领域之间建立起沟通的桥梁。借助本研究的成果, 金融分析领域的研究者可以对算法交易中常用的AI技术建立基本认识, 同时AI领域的专家也能借助本研究将他们的专业知识扩展到一个前景广阔的应用领域。


在上述研究动机和研究空隙的基础上,  本研究希望通过对当前算法交易中常用的AI技术进行汇总和分析, 并结合金融市场的特性回答以下两个相互关联的研究问题。第一个研究问题的重点在于, 通过对近5年来的AI在算法交易中应用的研究, 得出当前在不同金融市场广受关注的AI技术。对这个研究问题的回答, 能够为相关开发者和研究者在寻找适合不同金融市场的算法交易技术路线的过程提供明确的指引. 同时, 对第一个研究问题的分析过程同样对第二个研究问题的回答做了贡献。为了找出AI技术的引入所带来的机遇和挑战, 针对不同的AI技术的特点的分析是必不可少的。而由于不同的AI技术对研究者和开发者所具备的知识有不同的要求, 甚至实现不同AI技术所需的硬件也存在显著的差别, 因此不同的AI技术所能够带来的机遇和挑战是不能一概而论的。本研究致力于通过系统性文献研究同时兼顾AI的机遇和挑战, 从而帮助从业者们在性能和成本之间进行取舍。

RQ 1: "*Welche Methoden und Praktiken des algorithmischen Handelns existieren in der Literatur ?*

RQ2: *"Welche Herausforderungen und Chancen bietet die Integration von AI in bestehende algorithmische Handelssysteme?*




# 2 Theoretischer Hintergrund

不同于 1.2 中的概念, 我想在这里阐述位于第四章的文献中涉及的多种解决方案的理论基础。这些解决方案大致包括5个方面:
1. 高频交易
2. 机器学习
3. 深度学习
4. 强化学习
5. (深度强化学习)
6. 启发式算法

注意, 本章只涉及**不同模型和算法的基本原理(例如AT常用的机器学习算法原理)**, 目的是让读者更好地理解本研究的第四章中的研究成果。

Festlegung / Beschreibung des Review Scope:
Typ: Scoping Review
Ziel: eine fundierte theoretische Grundlage über KI & AT

## 2.1 高频交易
作为传统的算法交易方法, 需要被提及的原因是, 我需要回答第二个研究问题。

高频交易的原理以及为什么算法交易采用高频交易

举例论证, 对比个人投资者在一定时间段内做投资决策的方式和算法交易的差异。

高频交易的起源可以追溯到1998年(s-1-10, 7)。正如在第一章提到的, 高频交易被看作算法交易的一种特殊形式。算法交易通常被用于发达金融市场, 例如美国的股票市场或者其他发达的市场货币(s-1-10)。In fact, in 2009–2010, it was estimated that 60–70% of trading occurred is through HFT(s-1-10, 3). In a recent report, JP Morgan confirmed that about 60% of trading on exchanges is algorithmic (s-1-10, 4).

HFT is the use of computerized trading algorithms to buy and sell assets quickly and frequently, with a short holding period – to the single minute, second, or even millisecond level – to earn miniscule margins on each trade(s-1-10. 3 Hossain, 2022). 因此理想化的高频交易中, 动态交易策略通常被采用并且动态交易策略应该有能力根据金融市场的变化趋势自动被调整(s-1-9)。在学术界, 采用基于深度强化学习算法设计并优化采用动态交易策略的模型已经成为主流。Market-making, liquidity provision, and arbitrage opportunities in fragmented markets are key areas where HFT excels (3).




## 2.2 Machine Learning

分成小章节, 分类标准是 2.2.1 监督学习 和 2.2.2 无监督学习, 分别介绍文献中常用的方法中涉及的机器学习算法的原理:
- 随机森林
- XGBoost
- SVM
- ...

再加上这些算法各自的用途, 不仅局限在算法交易


提示静态交易

Machine learning (ML) has become a game-changer in algorithmic trading, allowing systems to learn from data and adapt to changing market conditions. ML-based approaches include various techniques such as supervised learning, unsupervised learning, and reinforcement learning. These algorithms analyze historical data, identify patterns, and make predictions about future price movements. The advantage of ML is its ability to uncover non-linear relationships and adapt to evolving market dynamics (3).

机器学习方法在量化金融领域被应用于创造更加精确的预测和提升金融模型的性能。 但是值得关注的是, 并不是所有的机器学习模型能够解决全部问题。具体的模型选择需要根据所解决的问题特征和可用的数据。因此machine learning and finance professionals should work together to obtain the best results. (2)

分类算法作为一种在金融应用中常用的技术, is supposed to categorize a certain stock as a “STRONG BUY” or a “STRONG SELL,” as well as “BUY,” “SELL,” and “HOLD.”(2). 一个常用的分类算法是朴素贝叶斯。 the procedure uses the Naive-Bayes classification algorithm that learns to predict the decision strategy once observed the set of predictors(s-1-1).
但是数据集里的不均匀分布的类别标签是分类任务变得困难(2), 因此如何解决这一问题并提升机器学习模型运行效率是亟待解决的问题。

- [x] 
Perhaps one of the simplest yet highly-effective techniques is known as Support Vector Machines (SVM) (Janardhanan et al., 2015; Pal & Mather, 2004). SVMs are used to identify the hyperplane that best separates a binary sampling. If we imagine a set of points mapped onto a 2d plane, the SVM will find the best line that divides the two different classifications in half. This technique can easily be expanded to work on higher-dimensional data, and since it is so simple, it becomes intuitive to see the reason behind a classification(s-1-3).
其他流行的分类方法有决策树和随机森林。这两种基于树的分类方法流行的原因在于其可解释性。Random Forest operates by constructing a multitude of decision trees at training time and outputting he classification of the individual trees (s-1-3, ).



与分类问题中, 类别标签这个离散的目标变量不同, 回归问题的目标变量是连续的。例如, 预测一项资产未来的价格就属于回归问题(2)。但是由于金融市场的复杂性, 使用简单的线性回归模型通常无法拟合多个自变量的回归问题。Ridge Regression is a model of approximating the coefficients of multiple regression frameworks in incidents where the independent variables are greatly correlated. Ridge regression is specifically instrumental for reducing the challenge of multicollinearity in linear regression, which mostly happens in algorithms with large quantities of parameters(1-3).


## 2.3 Deep Learning
同上, 常见的模型:
- MLP
- CNN
- RNN
- LSTM
- Transformer

再加上深度学习模型的常见的用途, 不仅仅局限在算法交易领域。

ANN
深度学习技术是一种机器学习的类型, 使用人工神经网络(ANNs). 人工神经网络由输入层、隐藏层、输出层组成, 见下图. 当一个ANN拥有多于一个隐藏层的时候, 就被称为Deep Neural Network (DNN), 也被称为Deep Feedforward Network (DFN) 或者 Multi-Layer Perceptron(MLP)。
Each artificial neuron has a weight and threshold and connects to others. Nodes deliver data to the next tier if their output exceeds the threshold. 为了模拟人脑, 人工神经网络的相邻层次之间互相连接(2 ..)。ANN的输入特征就是经过了预处理的关于金融市场的数据, 输出则是被期望的预测指标。deep learning combines features through multilayer network structures and nonlinear transformations with strong perception and representation capabilities(1-1). 
![[Pasted image 20240921175802.png]]

针对不同的问题类型, 深度学习提供了多种模型。在算法交易中, 常用的模型有以下几种:


CNN
卷积神经网络起初是为了处理图像和空间数据设计。CNN 通过卷积层提取特征，使用池化层减少特征的维度，并在最后使用全连接层进行分类或回归。卷积层可以捕获局部空间特征，而池化层有助于减少计算复杂度并防止过拟合。


RNN
RNNs have found success in various domains, particularly in time series forecasting, thanks to their significant predictive capabilities (1-5). 作为处理序列数据的神经网络，RNN最大的特点是具有循环结构，能够记忆先前的输入信息。在时间序列、自然语言处理等任务中，RNN 可以利用时间或顺序上的依赖关系，但传统的RNN存在梯度消失问题，限制了长序列的学习能力。

LSTM
LSTM 是为了解决 RNN 的梯度消失问题而提出的一种改进模型。LSTM 引入了“记忆单元”和“门”机制（输入门、遗忘门、输出门），可以有效捕捉长序列中的远程依赖关系，因此在处理长时间依赖的任务中表现优异，同时LSTM可以借助其内部状态, 即记忆 , 处理可变长度的输入序列(1-1)。在下图LSTM结构图中, 一个LSTM单元由细胞ct, 输入门it, 输出门ot, 遗忘门ft组成。The cell memorizes values at arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. (1-1)

![[Pasted image 20240922000026.png]](1-1)

当模型的输入特征例如时间序列, 是相同长度的, 那么从更多角度提取特征是可能的。例如, the output of a moment is not only related to the information of past moments, but also to the information of subsequent moments. 因此, 额外添加一个神经网络层用于反向传递信息从而提升模型的性能是可行的。实现这个功能的LSTM叫做双向LSTM。The bidirectional LSTM consists of two layers of LSTM, which have the same input but different directions of information transmission (1-1) [47].




Transformer
Transformer 是一种基于注意力机制的模型，它抛弃了传统的RNN结构，完全依赖自注意力机制来捕捉序列中的全局依赖关系。Transformer 的主要组成部分包括编码器-解码器结构、多头自注意力机制、位置编码等。它在自然语言处理任务中表现出色，尤其是在机器翻译和语言生成任务中。

## 2.4 Reinforcement learning
相比于 2.2 和 2.3 有什么特殊之处, 以及强化学习所能够解决的问题

强化学习算法的基本组成部分, 以及最常见的变体

机器学习和深度学习由于其训练过程的本质, 无法很好地适应动态的交易策略。而静态交易策略一旦被确定, 该策略在整个交易过程中将保持不变. 在金融市场的不确定性提升时, 静态交易策略将会面临极大的风险。因此, 一个能够根据金融市场环境变化而自我优化的交易策略是必要的。不同于前面提到的机器学习和深度学习方法, 强化学习能够通过智能体与环境的交互并借助奖惩机制实现交易策略的自我优化, 进而找到最优的动态交易策略  (s-1-9, 1-1)。

如下图所示的是一个包含单个智能体的强化学习模型。该模型包含两个部分, 分别是智能体和环境。As a result of the interaction between the agent and the environment in a recurring cycle, a chain of states, actions, and rewards is created. According to this framework, an intelligent agent performs an action (A t) concerning the current state at time (t) and receives a reward ( R t+1) as a result of the performed action. Then, the agent observes a new state (S t+1) and performs the next action based on the new state. This iterative framework continues until the agent converges to an optimal policy by maximizing a notion of cumulative reward (s-1-4).

当一个环境中同时存在多个智能体时, 深度学习框架与单个智能体类似。区别在于, 多个智能体之间也会进行交互, 这种交互可以是合作的或者竞争的(s-1-4 (Busoniu et al., 2006)。


![[Pasted image 20240921121323.png]]


一个特殊的变体是深度强化学习, Q function in deep reinforcement learning represents the expected cumulative reward that an agent can obtain by taking a particular action in a given state and following a particular strategy thereafter.Q networks are designed to learn an agent’s optimal action selection strategy by estimating the Q value (expected cumulative reward) for different state–action pair. The Q-Network structure is a key part of deep reinforcement learning algorithms, which has an important impact on the final results of the model (s-1-5).
Q-learning is a model-free off-policy RL method in which the agent aims to obtain the optimal state-action-value function by interacting with the environment. It maintains a state-action table Q[S, A] called as Q-table containing Q-values for every state-action pair (s-1-9).


## 2.5 启发式算法
适用范围最广: 遗传算法

用于解决优化问题。

## 2.6 LLM
并入深度学习


# 3 Methodik
![[Pasted image 20240819094147.png]]
Inklusionskriterien: Finance, AT, Machine Learning, AI, neuer als 2019
Exklusionskriterien: alt, Programmierung, Mathematik

文献检索过程:
[[Literatur Recherche - Protocol]]

从3个数据库中共检索出32篇文献。在第三章, 我需要写出检索时采用的字符串, 和两个数据库。
也要提及保留和删除的原则。

为了将检索出的文献进行更好的管理, 我会按照研究者在每篇论文中提出的方法在表格中分类标记。分类的标准是作者采用的技术路线, 即第二章提到的4种。此外, 我还会在表格中增加一列用于展示该研究涉及的具体投资市场, 例如股票, 外汇等。

统一在3个数据库中选择2019年以后的期刊文章, 因为GPT2模型在2019年2月被发明。尽管GPT2的基础模型Transformer早在2017年就被谷歌的几个工程师提出, 但是其大规模应用给科研领域带来的巨大影响发生于2019年以后。此后, 以Open AI的Chatgpt为代表的多种AI应用逐渐进入大众的视野, 并被应用于工业界。

## 3.1 Aufbau der Suchstrategie
Die Suchstrategie bildet eine wichtige Grundlage für das Suchen nach passender und relevanter Literatur. Im Folgenden wird der Prozess der Suche transparent dargelegt. Die Basis der Literaturrecherche bilden Datenbanken wissenschaftlicher Quellen. Da die Nutzung einer einzelnen Datenbank zu einseitig wäre und nicht das gesamte Forschungsspektrum abdecken könnte, bezog sich die Suche auf drei Datenbanken. Die in dieser Recherche verwendeten Datenbanken „Scopus“, „EBSCOhost“  und „DBLP“ wurden mit unterschiedlichen Strategien durchsucht, sodass ein einheitliches Vorgehen und eine Vergleichbarkeit gewährleistet werden konnten. Um diese Vergleichbarkeit herstellen zu können, wurde in allen Datenbanken mit den gleichen Suchstrings gearbeitet. Diese bestanden in ihrem Kern aus den allgemeinen Wörtern „learning analytics“, „education“ und „design“. Um die Übersicht zu wahren und die Suchergebnisse eingrenzen zu können, wurden die Suchen in den unterschiedlichen Datenbanken von Zeit zu Zeit spezifischer ausgerichtet und an den vorliegenden Forschungsbereich angepasst. Die Datenbanken unterscheiden sich in ihrer Möglichkeit, gewisse Suchfelder anzupassen und die Suche mittels spezieller Filterfelder zu konfigurieren. So wurde in den Datenbanken „Scopus“, „Web of Science“  und „Science Direct“ mit der Filtermaske „Article Title, Abstract and Keywords“ und in den Datenbanken sowie „EBSCOhost“ mit der Filtermaske „All Fields“ gesucht. Die Suchergebnisse in allen vier Datenbanken, basieren zum größten Teil auf englischsprachigen Fachartikeln, sodass die Arbeit mit deutschen Begriffen bzw. der deutschen Übersetzung der jeweiligen Suchtexte keine weiteren passenden Ergebnisse bringen. Deswegen wurden Suchen mit deutschen Wörtern im Fortlaufenden nicht berücksichtigt.





# 4 Ergebnisse
- Die zur Beantwortung der Forschungsfragen verwendeten Literatur und theoretischen Grundlagen.
- Ich hoffe, aus den Referenzen die Verbindungen zwischen verschiedenen Forschungsfragen zu identifizieren und diese Fragen in einer spezifischen logischen Reihenfolge zu arrangieren.
- Beachtung der Kombination verschiedener Autorenmeinungen.

- [ ] 思考如何组织这部分的具体内容, 
	- [ ] 1. 我想按照不同的投资领域, 例如股票, 大宗商品, 外汇, 加密货币, 投资组合等划分小标题
	- [ ] 2. 也可以按照不同的技术路线, 例如~~统计, 算法,~~ 机器学习, 深度学习, 深度强化学习

?? 我更倾向于采用第二种方案, 因为我的研究等题目是关于AI技术的, 所以在这一章也采用技术路线作为标题是合理的。标题结构如下:

## 4.1 机器学习
分析检索到的文献中 , 哪些研究采用了机器学习.
其中, 有些学者仅仅采用机器学习的方法实现算法交易, 这是因为不同学者对于算法交易有不同的理解。例如有人认为只要满足高频交易并仅仅通过ML模型辅助传统的HFT即可。但是也有学者认为, 我们不应该仅仅使用预测性分析来预测市场或者某只股票的未来走势, 而是要采用更聪明的方式, 即得到更好的投资策略。

特点和优势

每个子章节用于概括文献中使用机器学习技术的文献所涉及的投资市场

值得注意的是, 机器学习通常被用于预测某种资产未来的价格走势, 但是无法直接输出潜在的投资决策。

### 4.1.1 股票市场


### 4.1.2 加密货币




## 4.2 深度学习

采用深度学习的学者更多, 尤其是LSTM的多种变体

还有一种和生成式AI强相关的方法: Transformer, 但是直接相关的文献只有一篇

特点和优势

## 4.3 强化学习
本研究的重点, 也是当前算法交易领域最热门的主题
我检索到的文献中, 采用强化学习和深度强化学习的文章数量最多

目前的研究成果中, 应用最广且看上去效果最好的一种方法。但是同时也是最复杂的

强化学习也被更加广泛地应用于多种金融市场上的算法交易:

### 4.3.1 股票
### 4.3.2 期货
### 4.3.3 加密货币
### 4.3.4 外汇


## 4.4 优化算法
例如遗传算法等启发式算法在前面三种方法的基础上进行优化, 

或者采用人工校正, 从而避免AI造成的过拟合问题, 进一步提升投资决策的收益率

### 4.4.1 启发式算法
启发式算法 例如遗传算法, 可以将人类专家的投资决策整合到AI的行为中。

### 4.4.2 人工校正



# 5 Diskussion

## 5.1 文献中的方法和实践
把第四章进行总结, 从而回答第一个研究问题: 
- *Welche Methoden und Praktiken des algorithmischen Handelns existieren in der Literatur ?*
并说明这些方法和实践的重要程度。

## 5.2 AI给算法交易带来的机遇和挑战
为了回答第二个研究问题, 我需要在第五章更进一步论述AI技术给算法交易系统带来的机遇和挑战, 可以使用第四章提到的方法和实践, 但是最好不要重复引用同一篇文献的同一个句子。
- *"Welche Herausforderungen und Chancen bietet die Integration von AI in bestehende algorithmische Handelssysteme?*

## 5.3 反省和展望
接下来的研究应该区分广义上的AI和当下以大语言模型为基础的生成式AI在算法交易领域的应用。
