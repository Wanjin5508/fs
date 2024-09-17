
![[Pasted image 20240916162710.png]]

深度强化学习, [[s-1-4 A multi-agent deep reinforcement learning framework for algorithmic trading in financial markets.pdf]]

本文比 s-1-4 更完善, 采用了一个新的网络 Multi Agent Double Deep Q- Network, 能够兼顾利润最大化和风险管理 

创新点体现在, 本研究采用了两个不同的 agents , 分别是两个不同的时间序列特征提取网络, 即 TimesNet 和 Multi-Scale CNN

此外, 另一个创新点在于, 本研究提出的模型具备很好的泛化能力, 即采用预训练的方法, 在一个复合的数据集上对模型进行预训练, 模型学到的知识可以很好地被转移到其他领域的投资

---

intro中涉及的 AT的描述
传统投资方式, AT 的优势体现在哪些方面? --> 可以直接用

还提到了***缺点***

AT 的两种类别

相比于单个agent的优势

注意, 本文的多个智能体和 s-1-4 不同, 这里每个 agent 表示不同的投资偏好, 上篇指的是不同的领域知识


related works中提到了很多常见的用于提取时间序列中特征的方法, 例如多种神经网络的组合

居然提到了自注意力机制, 还有 timenet的模型架构
