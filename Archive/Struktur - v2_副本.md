![[Pasted image 20240813125052.png]]


注意这个版本是pdf基础上, 经过导师修改的新版, 在这里明确了具体的RQ以及文章的结构。[[会议记录&邮件往来|导师的反馈]] 

导师说, 结构没什么大问题, 不太需要修改

既然是突出AI技术对AT的影响, 那么我希望按照不同的AI技术划分第四章 Ergebnisse 的小标题, 目前计划:
- 统计方法
- 机器学习
- 深度学习
- 强化学习
- (深度强化学习)

# 1 Einleitung
## Motivation
- Die Einführung von algorithmischem Handel (AT) im traditionellen Finanzsektor bietet neue Investitionsmöglichkeiten und ~~schafft höhere Investitionsrenditen.~~。(这个不好, 因为我们需要的是清晰的陈述。)
- Die Entwicklung der KI führt zu Veränderungen in verschiedenen Branchen.
- Diese Arbeit untersucht die vielfältigen Auswirkungen dieser neuen Technologie auf den algorithmischen Handel.

>Um die Forschungslücke zu schließen, sind weitere Literaturstudien erforderlich. Welche spezifischen Aspekte der Forschungslücke sollten jedoch aufgelistet werden? 
>
>Ich möchte aus einer vergleichenden Forschungsperspektive Forschungslücken identifizieren, insbesondere durch die Kombination verschiedener praktischer Fallstudien, um die Lücke in der vergleichenden Forschung über die tatsächliche Leistung von traditioneller KI und generativer KI im algorithmischen Handel (AT) zu füllen.

- [ ] 论文的动机需要经过文献检索以后, 综合当前研究现状
- [ ] 当然也可以加入我个人的动机 --> 交叉学科winfo, 希望找到两个领域 的合适的结合
- [ ] 研究空隙: 生成式AI兴起以来, 各行各业的常规工作流都收到了或多或少的影响。目前还没有研究人员深入研究不同的AI技术在宏观层面对算法交易对影响, 因为根据 文献检索的结果, 绝大多数的研究倾向于从数学和算法的角度对算法交易进行分析和优化。本研究希望在AI产品被大规模应用的背景下, 为其他研究者建立一个稳固的知识基础。借助本研究的贡献, 其他的研究者可以对AI技术影响下的算法交易迅速建立初步认识, 从而提升他们进一步研究的效率。

## RQ - Candidate

> [!3 RQs]
> 根据导师的反馈, 保留前三个研究问题即可。
> 导师提到可以将 2 和 3 合并, 但是我还是倾向于分开, 因为分开回答会更有针对性。尤其是考虑到在论文的Ergebnis部分我是需要按照某种逻辑顺序展示我的研究结果, 而不是完全按照 RQ 的顺序。
> 

###  **Forschungsfrage 1 (RQ 1)**: AT 方法和实践

Wie sehen die gängigen Arbeitsabläufe in den besten Praktiken des algorithmischen Handels aus?

--> 导师觉得这个问题非常 unklar, 合适的RQ需要明确的限定并且能够被准确地回答。

修改如下: 
"*Welche Methoden und Praktiken des algorithmischen Handelns existieren in der Literatur (für die Vorhersage von Marktbewegungen)“?*
(括号内的内容是optional, 用于限定我的研究内容在预测 Vorhersage), 但是我不希望添加这个限定, 因为我希望通*过多个方面的研究*来回答这个R Q

- [ ] --> 我需要研究当前科研和实践中的现状, 考虑不同的投资标的物/市场上, 算法交易采用的方法和实践
	- [ ] 这里要提一下传统的算法交易, 即基于预设规则。还有本文的重点: 基于广义AI的方法和实际, 例如机器学习, 深度学习和强化学习

###  **Forschungsfrage 2 (RQ 2)**: AI 机遇和挑战
Welche Vorteile bietet die Einführung von KI im Vergleich zu traditionellen, auf maschinellem Lernen basierenden algorithmischen Handelssystemen?
--> 导师觉得不错, 但是既然都研究 优势 了, 那就再添加一些客观的对比, 即加入**挑战和机会**, 这样能写的内容就变多了

修改如下: 
*"Welche Herausforderungen und Chancen bietet die Integration von AI in bestehende algorithmische Handelssysteme?*

###  **Forschungsfrage 3 (RQ 3)**: 对机构投资者对影响
适合作为 RQ , 但是Motivation中需要事先分析一下, 为什么我试图研究机构投资者, 而不是散户或者其他的市场参与者。以及, 为什么在我的研究中, 机构投资者扮演者更加重要的角色(可以从投资规模, 规模效应, 技术积累等角度)

*Welche Auswirkungen hat die Anwendung von KI auf Investitionen für institutionelle Investoren?*

> Die **RQ3** ist verständlich. Sie sollten dann nur in der ***Motivation/Einleitung*** erklären, warum Sie die Frage für institutionelle Investoren beantworten möchte und z.B. für andere Markteilnehmer nicht. Warum spielen gerade diese eine wichtige Rolle? 



###  ~~**Forschungsfrage 4 (RQ 4)**:~~ 
~~Welche Veränderungen ergeben sich aus systemarchitektonischer Sicht nach der Einführung von KI in Systemen für algorithmischen Handel?~~

--> 对于 FS来说太难了

## Forschungsbeitrag
Der erwartete Beitrag dieser Studie besteht darin, durch eine systematische Literaturübersicht die oben genannten (zwei oder drei) potenziellen Forschungsfragen zu beantworten. Dies soll anderen Forschern und Entwicklern eine。*fundierte theoretische Grundlage* bieten.
--> 导师说, 理论基础在FS中仅仅扮演一个次要的角色, 因为 FS 的目的是从 RQ 中获取 ***Erkenntnisgewinnung***. 即einen aktuellen Stand der Forschung zu dem Thema erarbeiten


# 2 Theoretischer Hintergrund
## Definition
- Unterschied zwischen traditioneller Finanzwirtschaft und AT
- Definition und Entwicklung von Algorithmic Trading sowie übliche Implementierungen
- Gen AI vs. traditional AI (Machine Learning)
- Weitere Konzepte zu KI und Investitionen

- [ ] 核心定义区分金融领域和计算机领域

## Festlegung / Beschreibung des Review Scope
Typ: Scoping Review
Ziel: eine fundierte theoretische Grundlage über KI & AT

# 3 Methodik
![[Pasted image 20240819094147.png]]
Inklusionskriterien: Finance, AT, Deep Learning, AI, neuer als 2021
Exklusionskriterien: alt, Programmierung, Mathematik

# 4 Ergebnisse
- Die zur Beantwortung der Forschungsfragen verwendeten Literatur und theoretischen Grundlagen.
- Ich hoffe, aus den Referenzen die Verbindungen zwischen verschiedenen Forschungsfragen zu identifizieren und diese Fragen in einer spezifischen logischen Reihenfolge zu arrangieren.
- Beachtung der Kombination verschiedener Autorenmeinungen.

- [ ] 思考如何组织这部分的具体内容, 
	- [ ] 1. 我想按照不同的投资领域, 例如股票, 大宗商品, 外汇, 加密货币, 投资组合等划分小标题
	- [ ] 2. 也可以按照不同的技术路线, 例如~~统计, 算法,~~ 机器学习, 深度学习, 深度强化学习


# 5 Diskussion






