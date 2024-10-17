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

# 1 Einleitung

## 1.1 Motivation
- Die Entwicklung der KI führt zu Veränderungen in verschiedenen Branchen.
- Diese Arbeit untersucht die vielfältigen Auswirkungen der KI-Technologie auf den algorithmischen Handel.

Die Forschungslücke: Seit dem Aufstieg generativer KI hat sie die regulären Arbeitsabläufe in verschiedenen Branchen beeinflusst. Derzeit gibt es nur wenige Untersuchungen, die sich mit den allgemeinen Auswirkungen von KI auf den algorithmischen Handel befassen, insbesondere im aktuellen Trend der generativen KI. Die meisten Studien neigen dazu, den algorithmischen Handel aus mathematischer und algorithmischer Sicht zu analysieren und zu optimieren und neue Handelsmodelle zu entwickeln. Viele wissenschaftliche Arbeiten konzentrieren sich nur auf das theoretische Wissen der vorgeschlagenen Modelle, wie z.B. die Vor- und Nachteile einzelner maschineller Lernalgorithmen.

Da der algorithmische Handel ein interdisziplinäres Feld ist, das hohe Anforderungen an Mathematik und Computeralgorithmen stellt, schafft er für Forscher aus unterschiedlichen Branchen eine hohe Einstiegshürde. Diese Studie zielt darauf ab, anderen Forschern eine solide Wissensbasis zu bieten, indem sie den Einfluss der KI-Technologie auf den algorithmischen Handel untersucht.

## 1.2 Zentrale Begriffe
In diesem Abschnitt werden die gängigen Begriffe der Studie detailliert erklärt und unterschieden, damit Leser aus verschiedenen Forschungsbereichen die Bedeutung der verschiedenen Fachbegriffe korrekt verstehen. Diese Begriffe werden nach Fachgebieten in zwei Kategorien eingeteilt: Begriffe im Zusammenhang mit Finanzmärkten und Begriffe im Zusammenhang mit KI-Technologie. (Dieser Abschnitt kann im Laufe der Forschung aktualisiert werden.)

### 1.2.1 Finanzmarkt
Einschließlich:
- Definition und Entwicklung des algorithmischen Handels, FinTech
- Häufige Finanzmärkte, wie Aktien, Devisen, Futures, Kryptowährungen...
- Fachbegriffe zur Beschreibung des Finanzhandels
- Indikatoren zur Messung der Anlageerträge, wie z.B. der Sharpe Ratio

### 1.2.2 AI Technology
- Derzeit bezieht sich der enge Begriff der KI auf generative KI oder AGI, während der allgemeine Begriff der KI maschinelles Lernen, Deep Learning und Reinforcement Learning umfasst. In diesem Artikel werde ich die allgemeine Definition von KI verwenden, da generative KI noch nicht weit verbreitet im algorithmischen Handel eingesetzt wird.
- Definition des maschinellen Lernens
- Definition des Deep Learnings und der Unterschiede zum maschinellen Lernen
- Definition des Reinforcement Learnings und Unterschiede zu Deep Learning
- Diese drei Techniken lösen unterschiedliche Arten von Problemen, sodass kein Modell das "beste" ist. Forscher müssen das am besten geeignete Modell für das jeweilige Projekt auswählen.

## 1.3 Forschungsbeitrag
Der erwartete Beitrag dieser Studie besteht darin, durch eine systematische Literaturübersicht die oben genannten (zwei oder drei) potenziellen Forschungsfragen zu beantworten und so anderen Forschern eine fundierte theoretische Grundlage zu bieten.

# 2 Theoretischer Hintergrund

Anders als in Abschnitt 1.2, in dem Begriffe definiert werden, werde ich hier die theoretischen Grundlagen der in Kapitel 4 beschriebenen Lösungsansätze erläutern. Diese Lösungsansätze umfassen im Wesentlichen fünf Bereiche:
1. Hochfrequenzhandel (HFT)
2. Maschinelles Lernen
3. Deep Learning
4. Reinforcement Learning
5. (Deep Reinforcement Learning)
6. Heuristische Algorithmen

Wichtig: In diesem Kapitel werden nur die grundlegenden Prinzipien der verschiedenen Modelle und Algorithmen behandelt, um den Lesern ein besseres Verständnis der in Kapitel 4 präsentierten Ergebnisse zu ermöglichen.

Festlegung/Beschreibung des Review-Umfangs:
- Typ: Scoping Review
- Ziel: Eine fundierte theoretische Grundlage über KI und algorithmischen Handel zu schaffen.

## 2.1 Hochfrequenzhandel (HFT)
Als traditionelle Methode des algorithmischen Handels muss der Hochfrequenzhandel erwähnt werden, da ich damit die zweite Forschungsfrage beantworten möchte.

Hierbei werden die Prinzipien des Hochfrequenzhandels sowie die Gründe erläutert, warum der algorithmische Handel auf HFT zurückgreift.

Anhand von Beispielen wird der Unterschied zwischen den Investitionsentscheidungen eines individuellen Investors und dem algorithmischen Handel über einen bestimmten Zeitraum aufgezeigt.

## 2.2 Maschinelles Lernen
Dieser Abschnitt ist in Unterkapitel unterteilt, wobei das Klassifikationskriterium **2.2.1 Supervised** und **2.2.2 Unsupervised** ist. Es werden die in der Literatur häufig verwendeten Algorithmen des maschinellen Lernens erläutert:
- Random Forest
- XGBoost
- SVM
- ...

Zusätzlich wird auf die jeweiligen Einsatzgebiete dieser Algorithmen eingegangen, die sich nicht nur auf den algorithmischen Handel beschränken.

## 2.3 Deep Learning
Ähnlich wie im vorherigen Abschnitt werden hier die gängigen Modelle vorgestellt:
- MLP (Mehrschichtige Perzeptrons)
- CNN (Convolutional Neural Networks)
- RNN (Recurrent Neural Networks)
- LSTM (Long Short-Term Memory)
- Transformer

Zusätzlich werden die häufigen Anwendungsbereiche von Deep-Learning-Modellen beschrieben, die über den algorithmischen Handel hinausgehen.

## 2.4 Reinforcement Learning
Es wird herausgearbeitet, welche besonderen Eigenschaften Reinforcement Learning im Vergleich zu 2.2 und 2.3 hat und welche Probleme mit Reinforcement Learning gelöst werden können.

Grundlagen der Reinforcement-Learning-Algorithmen sowie deren häufigsten Varianten werden ebenfalls beschrieben.

## 2.5 Heuristische Algorithmen
Hier wird der am weitesten verbreitete Algorithmus, der genetische Algorithmus, behandelt.

Er wird zur Lösung von Optimierungsproblemen verwendet.



# 3 Methodik

Aus zwei Datenbanken wurden insgesamt 32 relevante Studien identifiziert. In diesem Kapitel werde ich die bei der Recherche verwendeten Suchstrings und die beiden Datenbanken angeben.

Außerdem werden die Auswahl- und Ausschlusskriterien erläutert.

Um die identifizierten Studien besser zu verwalten, werde ich die Methoden, die von den Forschern in jeder Studie verwendet werden, in einer Tabelle kategorisieren. Die Klassifizierungsstandards basieren auf den technischen Ansätzen, die im zweiten Kapitel beschrieben wurden (z. B. maschinelles Lernen, Deep Learning, Reinforcement Learning und heuristische Algorithmen). 

Zusätzlich werde ich eine Spalte in der Tabelle hinzufügen, um den spezifischen Investitionsmarkt darzustellen, auf den sich die jeweilige Studie bezieht, wie z. B. Aktien, Devisen usw.


# 4 Ergebnisse
Da sich diese Arbeit auf die Auswirkungen von KI-Technologien auf den algorithmischen Handel konzentriert, werde ich die Kapitelüberschriften nach den verschiedenen KI-Technologien aufteilen. Geplant sind folgende Unterkapitel:
- Statistische Methoden
- Maschinenlernen
- Deep Learning
- Reinforcement Learning
- (Deep Reinforcement Learning)

## 4.1 Maschinenlernen
Analyse der in der Literatur verwendeten Studien, die maschinelles Lernen im algorithmischen Handel einsetzen.

Manche Forscher setzen maschinelles Lernen ausschließlich zur Umsetzung des algorithmischen Handels ein, da es unterschiedliche Auffassungen über den algorithmischen Handel gibt. Einige sind der Meinung, dass es ausreicht, den Hochfrequenzhandel (HFT) zu erfüllen und traditionelle HFT-Strategien lediglich durch ML-Modelle zu unterstützen. Andere wiederum argumentieren, dass man nicht nur prädiktive Analysen verwenden sollte, um den zukünftigen Markt- oder Aktienkurs vorherzusagen, sondern intelligentere Ansätze verfolgen muss, um bessere Anlagestrategien zu entwickeln.

Jedes Unterkapitel fasst die in der Literatur beschriebenen Anlagemärkte zusammen, in denen maschinelle Lerntechniken angewendet wurden.

Es ist erwähnenswert, dass maschinelles Lernen häufig verwendet wird, um den zukünftigen Preisverlauf eines Vermögenswerts vorherzusagen, jedoch keine direkten Handlungsempfehlungen für potenzielle Anlageentscheidungen liefert.

### 4.1.1 Aktienmarkt
Untersuchung, wie maschinelles Lernen im Aktienmarkt eingesetzt wird.

### 4.1.2 Kryptowährungen
Untersuchung des Einsatzes von maschinellem Lernen im Bereich der Kryptowährungen.

## 4.2 Deep Learning
Hier werde ich die Forschung analysieren, die Deep-Learning-Methoden verwendet. Besonders viele Studien beschäftigen sich mit LSTM und seinen Varianten.

Eine weitere Methode, die eng mit generativer KI verbunden ist, ist der Transformer, aber es gibt nur eine direkt damit in Zusammenhang stehende Studie.

## 4.3 Reinforcement Learning
Schwerpunkt der Arbeit und derzeit eines der heißesten Themen im algorithmischen Handel. Viele Studien verwenden Reinforcement Learning und Deep Reinforcement Learning.

Unter den bisherigen Forschungsergebnissen ist dies eine der am weitesten verbreiteten und vielversprechendsten Methoden. Gleichzeitig ist sie jedoch auch die komplexeste.

### 4.3.1 Aktien
Verwendung von Reinforcement Learning im Aktienmarkt.

### 4.3.2 Futures
Verwendung von Reinforcement Learning im Futures-Markt.

### 4.3.3 Kryptowährungen
Verwendung von Reinforcement Learning im Bereich der Kryptowährungen.

### 4.3.4 Forex
Verwendung von Reinforcement Learning im Devisenmarkt.

## 4.4 Optimierungsalgorithmen
Zusätzlich zu den genannten KI-Methoden werden auch heuristische Algorithmen wie genetische Algorithmen eingesetzt, um die Handelsstrategien weiter zu optimieren.

### 4.4.1 Heuristische Algorithmen
Genetische Algorithmen und andere heuristische Methoden, die in Kombination mit maschinellem Lernen, Deep Learning oder Reinforcement Learning verwendet werden.

### 4.4.2 Manuelle Anpassungen
Manuelle Korrekturen zur Vermeidung von Overfitting und Verbesserung der Rentabilität von Handelsentscheidungen.

# 5 Diskussion

## 5.1 Methoden und Praktiken in der Literatur
Zusammenfassung der in Kapitel 4 diskutierten Methoden und Praktiken zur Beantwortung der ersten Forschungsfrage.
- RQ 1: *Welche Methoden und Praktiken des algorithmischen Handelns existieren in der Literatur ?*
## 5.2 Chancen und Herausforderungen durch AI
Um die zweite Forschungsfrage zu beantworten, muss ich im fünften Kapitel genauer auf die Chancen und Herausforderungen eingehen, die die Integration von KI-Technologien in algorithmische Handelssysteme mit sich bringt. Dabei können die in Kapitel vier erwähnten Methoden und Praktiken verwendet werden, allerdings sollte möglichst vermieden werden, denselben Satz aus derselben Quelle erneut zu zitieren.

- RQ 2: _"Welche Herausforderungen und Chancen bietet die Integration von KI in bestehende algorithmische Handelssysteme?"_

## 5.3 Reflexion und Ausblick

HFT(or AT) has become an important part of futures markets around the globe, and its role is growing with advancing technology for exchanges and traders, especially for the Asia‐Pacific markets(1-7). 

Ausblick auf zukünftige Forschungen, wobei zwischen allgemeiner KI und generativer KI (basierend auf großen Sprachmodellen) unterschieden wird.
