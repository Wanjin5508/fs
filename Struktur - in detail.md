![[Pasted image 20240813125052.png]]


# 0 Verzeichnis
[[#1 Einleitung]]
- [[#]]


[[#2 Theoretischer Hintergrund]]



[[#3 Methodik]]

[[#4 Ergebnisse]]
既然是突出AI技术对AT的影响, 那么我希望按照不同的AI技术划分第四章 Ergebnisse 的小标题, 目前计划:
- 统计方法
- 机器学习
- 深度学习
- 强化学习
- (深度强化学习)

[[#5 Diskussion]]


# 1 Einleitung
## 1.1 Motivation

- Die Entwicklung der KI führt zu Veränderungen in verschiedenen Branchen.
- Diese Arbeit untersucht die vielfältigen Auswirkungen der KI-Technologie auf den algorithmischen Handel.


研究空隙: 生成式AI兴起以来, 各行各业的常规工作流都受到了或多或少的影响。目前针对广义上的AI在算法交易中的影响的研究数量较少, 尤其是在当前生成式AI流行的趋势下。 根据文献检索的结果, 绝大多数的研究倾向于从数学和算法的角度对算法交易进行分析和优化, 并进一步开发新的算法交易模型 。大量学者的论文中只涉及他们提出的模型相关的理论知识, 例如采用传统机器学习方法进行算法交易模型设计的学者只会列出多种机器学习算法的优劣.
而算法交易作为一个跨学科且对数学和计算机算法有着较高要求的学科, 给来自不同行业背景的研究者和初学者带来了较高的进入门槛。因此, 本研究希望在AI产品被大规模应用的背景下, 为其他研究者建立一个稳固的知识基础。借助本研究的贡献, 其他的研究者可以对多年以来在AI技术影响下的算法交易方法和金融市场迅速建立初步认识, 从而提升他们进一步研究的效率。


## 1.2 Zentrale Begriffe
在这一节, 本研究相关的常用概念被详细阐释和区分, 从而让来自不同研究领域的读者们准确理解不同专业术语的含义。我将会将这些概念按照学科领域不同分成两类, 即金融市场相关的概念和AI技术相关的概念。
(这部分可以随着研究的深入不断更新)

### 1.2.1 Finanzmarkt
包括:
- 算法交易的定义, 发展历程, FinTech
- 常见的金融交易市场, 例如股票, 外汇, 期货, 加密货币......
- 描述金融交易的专业术语, 上面提到的投资标的物的基本原理
- 用于衡量不同资产的投资回报率的指标, 例如最常见的是夏普比率
- 


### 1.2.2 AI Technology
- 目前狭义上的AI指的是生成式AI, 或者AGI, 而广义上的AI包括了机器学习, 深度学习和强化学习算法在训练数据集上训练好的模型。在算法交易领域, 目前尚未有生成式AI大规模的应用, 因此我选择在本文中采用广义的AI定义。事实上大多数学者的研究也是这么做的 
- 机器学习的定义
- 深度学习的定义, 与机器学习的区别
- 强化学习的定义, 与深度学习的区别。
- 三种技术分别适合解决的问题类型不同, 因此不存在最好的模型, 研究者需要根据实际项目选择最合适的



## 1.3 Forschungsbeitrag
Der erwartete Beitrag dieser Studie besteht darin, durch eine systematische Literaturübersicht die oben genannten (zwei oder drei) potenziellen Forschungsfragen zu beantworten. Dies soll anderen Forschern und Entwicklern eine *fundierte theoretische Grundlage* bieten.
--> 导师说, 理论基础在FS中仅仅扮演一个次要的角色, 因为 FS 的目的是从 RQ 中获取 ***Erkenntnisgewinnung***. 即einen aktuellen Stand der Forschung zu dem Thema erarbeiten

RQ 1: "*Welche Methoden und Praktiken des algorithmischen Handelns existieren in der Literatur ?*

- [ ] --> 我需要研究当前科研和实践中的现状, 考虑不同的投资标的物/市场上, 算法交易采用的方法和实践
	- [ ] 这里要提一下传统的算法交易, 即基于预设规则。还有本文的重点: 基于广义AI的方法和实际, 例如机器学习, 深度学习和强化学习, 并在第四章对这些不同的技术路线分别进行分析

RQ2: *"Welche Herausforderungen und Chancen bietet die Integration von AI in bestehende algorithmische Handelssysteme?*
原本关于机构投资者的 RQ 3 由于直接相关的文献数量过少, 因此我认为不适合作为一个单独的研究问题。将原本的RQ 3 放到动机部分或者末尾的展望部分更加合适。 

在这个系统性文献研究中, 我期望实现的贡献是通过对学术界关于算法交易的最新研究成果以及最佳时间的分析, 为其他研究者提供关于该领域中常用的解决方案和实现这些方案所需要的知识基础。本研究还希望通过对其他研究者提出的理论和进行的实践的总结, 讨论AI的引入所带来的机遇和挑战, 这对于市场上的参与者们有不同的意义

# 2 Theoretischer Hintergrund

不同于 1.2 中的概念, 我想在这里阐述位于第四章的文献中涉及的多种解决方案的理论基础。这些解决方案大致包括5个方面:
1. 高频交易
2. 机器学习
3. 深度学习
4. 强化学习
5. (深度强化学习)
6. 启发式算法

注意, 本章只涉及不同模型和算法的基本原理, 目的是让读者更好地理解本研究的第四章中的研究成果。

Festlegung / Beschreibung des Review Scope:
Typ: Scoping Review
Ziel: eine fundierte theoretische Grundlage über KI & AT

## 2.1 高频交易
作为传统的算法交易方法, 需要被提及的原因是, 我需要回答第二个研究问题。

高频交易的原理以及为什么算法交易采用高频交易

举例论证, 对比个人投资者在一定时间段内做投资决策的方式和算法交易的差异。

## 2.2 Machine Learning

分成小章节, 分类标准是 2.2.1 监督学习 和 2.2.2 无监督学习, 分别介绍文献中常用的方法中涉及的机器学习算法的原理:
- 随机森林
- XGBoost
- SVM
- ...

再加上这些算法各自的用途, 不仅局限在算法交易

## 2.3 Deep Learning
同上, 常见的模型:
- MLP
- CNN
- RNN
- LSTM
- Transformer

再加上深度学习模型的常见的用途, 不仅仅局限在算法交易领域。

## 2.4 Reinforcement learning
相比于 2.2 和 2.3 有什么特殊之处, 以及强化学习所能够解决的问题

强化学习算法的基本组成部分, 以及最常见的变体


## 2.5 启发式算法
适用范围最广: 遗传算法

用于解决优化问题。


# 3 Methodik
![[Pasted image 20240819094147.png]]
Inklusionskriterien: Finance, AT, Machine Learning, AI, neuer als 2019
Exklusionskriterien: alt, Programmierung, Mathematik

文献检索过程:
[[Literatur Recherche - Protocol]]

从两个数据库中共检索出32篇文献。在第三章, 我需要写出检索时采用的字符串, 和两个数据库。
也要提及保留和删除的原则。

为了将检索出的文献进行更好的管理, 我会按照研究者在每篇论文中提出的方法在表格中分类标记。分类的标准是作者采用的技术路线, 即第二章提到的4种。此外, 我还会在表格中增加一列用于展示该研究涉及的具体投资市场, 例如股票, 外汇等。

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
