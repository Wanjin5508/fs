# 1
- Herzlich Willkommen zu meiner Präsentation über das Thema “The Impact of AI Technology on Algorithmic Trading”

- Ich freue mich, Ihnen heute meine Vorgehensweise, Ergebnisse sowie Gedanken zu diesem Thema vorstellen zu dürfen.


# 2
-Ich habe meinen Vortrag in 4 Teile gegliedert.
-Zuerst spreche ich über den Forschungshintergrund und die Methodik.Inklusive eine kurze Motivation und die Vorgehensweise der systematischen Literaturrecherche
-Dann komme ich zu meinen Forschungsergebnissen, wobei die in der Literatur auftauchenden KI Technologien präsentiert werden.
-Im dritten Teil befasse ich mich dann mit dem Forschungsbeitrag und Ausblick.
-Zum Ende kommt die Diskussionsrunde.


# 3 
Ich komme jetzt zum ersten Teil, Forschungshintergrund und Methodik

# 4
Am Anfang möchte ich Sie in das Thema kurz motivieren. Seit dem Aufstieg der generativen KI, wie ChatGPT von OpenAI, sind zahlreiche Bereiche der Gesellschaft verändert worden. Diese erstaunlichen Fähigkeiten von KI haben Interesse von Forschenden mit unterschiedlichem Hintergrund geweckt, damit die Effizienz nicht nur in Forschung sondern auch in Industrie gestiegen werden kann.

Ich möchte jedoch besonders darauf hinweisen, dass Gen ai repräsentiert nicht das gesamte Spektrum der KI-Technologien. Damit komme ich zum nächsten Punkt.

# 5
- Auf dieser Folie sehen Sie eine Differenzierung der Fachbegriffe.

-Der Begriff der KI umfasst alle Technologien, die es Computern ermöglichen, menschliches Verhalten nachzuahmen und Entscheidungen zu treffen。

-Es ist durch das Schaubild zu veranschaulichen, dass der Begriff KI eine breite Bedeutung hat. Und Gen KI bildet nur eine Teilmenge von KI.

Im dritten Teil werde ich Ihnen ausführlich erläutern, wie diese KI-Technologien in der Literatur modifiziert, integriert und auf verschiedene algorithmische Handelsstrategien angewendet wurden.


# 6
-Soweit zum Fachbegriff. Ich wende mich nun dem Forschungsdesign zu. Bei der Bearbeitung habe ich anhand des Rahmenwerks für Literaturanalyse von Brocke verfolgt.

-Erstens, die Festlegung des Umfangs. Ich möchte eine umfassende Übersicht über die Anwendung von KI, inkl. ML, DL… in AH bekommen, deswegen habe ich den Umfang nicht auf einem spezifischen Finanzmarkt beschränkt.

-Zweitens, durch die Konzeptualisierung des Themenbereichs habe ich mein Verständnis über AH und KI vertieft, was eine gute Grundlage für die nächsten 2 Schritte bildete.

-Drittens, Literatursuche wurde in drei unterschiedlichen Datenbanken durchgesetzt. Ich werde Detail in nächste Folie zeigen.

-Viertens, die gefundenen Quellen wurden weiter analysiert und eine eigene Zusammenfassung wurde für jede ausgewählte Quelle erstellt.

-Am Ende wurde eine Forschungsagenda formuliert, anhand der Forschungsagenda wurde jeder Teil der Arbeit Schritt für Schritt fertiggestellt.


# 7
Um den Einfluss von KI-Technologien auf den algorithmischen Handel zu untersuchen, habe ich mit der Methode der systematischen Literaturrecherche relevante Arbeiten aus drei verschiedenen Datenbanken ausgewählt.

RQ1 RQ2

Um eine breitere Perspektive zu gewinnen, habe ich die Literaturrecherche ausschließlich mit den Suchbegriffen durchgeführt, die sich auf die erste Forschungsfrage beziehen. Der Grund dafür ist, dass in diesen drei Datenbanken nur wenige Forscher die Chancen und Risiken von KI eingehend analysiert haben.

Ein weiterer Grund ist, dass die Forscher von KI-Modellen in ihren Studien oft die Vor- und Nachteile der entsprechenden Technologien erwähnen.

Das rechte Schaubild veranschaulicht den Prozess der Literatursuche. Die 3 Datenbanken liefern insgesamt 82 einzigartige Publikationen. Nach dem Titel- und Abstractscreening habe ich noch 34 Quellen. Durch eine Volltextprüfung sind noch 27 Quellen beibehalten. Während der Bearbeitung sind 6 weitere Referenzen hinzugefügt.



# 8
Ich komme jetzt zu dem zweiten Teil, die Forschungsergebnisse.

# 9
In diesem Teil beginne ich mit der ML-Methode.

Manche Forscher vertreten die Ansicht, dass relevante Indikatoren des Finanzmarktes mithilfe von ML erfasst werden können. Durch die Analyse der Indikatoren kann der künftige Trend auf einem Finanzmarkt  vorhergesagt werden. Aber mit Berücksichtigung der komplexen Natur auf dem Finanzmarkt nimmt die Schwierigkeit der Marktprognose zu.

Klassifik:

Gegenstand: Marktrichtung

Richtungsorientierter Handel à eine Art des AH, bedeutet, dass Anleger basierend auf ihren Erwartungen an die Preisentwicklung des Marktes handeln

Wenn ein Händler beispielsweise einen Anstieg durch das Ergebnis der Klassifikation der zukünftigen Preise erwartet, kauft er Vermögenswerte, um sie bei einem Preisanstieg zu verkaufen und Gewinne zu erzielen (sog. Long-Position).

算法有三种, 这三种算法通常使用多种技术指标作为独立变量。

Regression:

Gegenstand: Volatilität auf dem Bitcoin-Futures Markt.

Aber nicht direkt messbar, à Proxy

Investoren können entsprechend ihrer individuellen Risikobereitschaft investieren.

# 10 
Annahme:

d.h. das aktuelle Muster einer Handelssitzung kann dem Muster einer Sitzung in der Vergangenheit ähneln. Daher sollten Anleger in der Lage sein, die optimale Investitionsentscheidung aus einer früheren ähnlichen Situation zu identifizieren und sie auf eine aktuelle, ähnliche Situation anzuwenden

Die Daten historischer Handelssitzungen können daher in N Gruppen oder Cluster unterteilt werden. Zwei Handelssitzungen in derselben Gruppe teilen bestimmte Merkmale, während Sitzungen in unterschiedlichen Gruppen Unterschiede aufweisen.

Diese Gruppen oder Cluster werden aus den historischen Sitzungsdaten mithilfe des K-Means-Clusterings gebildet

# 11
Im Vergleich zu traditionellen maschinellen Lernmethoden neigen Forscher dazu, Deep-Learning-Modelle zu verwenden, um abstrakte Informationen aus Finanzdaten zu extrahieren. In den recherchierten Publikationen sind CNN und LSTM die bei den Forschern beliebtesten Deep-Learning-Modelle. 

In der vorherigen Sektion erwähnte maschinelle Lernmethoden nutzen Zeitreihendaten zur Vorhersage von Markttrends. Allerdings ist die Repräsentationsfähigkeit von Zeitreihendaten fragwürdig, da sie lediglich die zeitliche Entwicklung des Zustands einer einzelnen Aktie zusammenfassen, ohne die Beziehungen zwischen verschiedenen Aktien zu berücksichtigen. Im Bereich der Informatik eignen sich Graphen am besten zur Darstellung von Beziehungsinformationen. Daher können die potenziellen Beziehungen zwischen verschiedenen Aktien durch gerichtete Graphen dargestellt werden.

Ein Vorteil dieses gerichteten Graphen liegt darin, dass er nicht nur die Korrelation zwischen verschiedenen Aktien quantifizieren kann, sondern auch durch die im Graphen durch die dargestellten Gewichte den Einfluss des historischen Preises einer Aktie auf den aktuellen Preis einer anderen Aktie ausdrückt.  

需要说明的是, 尽管CNN模型通常被应用于图像处理, 但是实际上CNN可以被应用于任何二维数据的处理和分析中。这是因为在计算机视觉领域, 图像数据本质上被当作包含三个通道的二维矩阵处理。因此有向图也可以被转换成二维的非对称矩阵, 从而被CNN模型接收。Daher lässt sich behaupten, dass sie unterschiedliche Aspekte der Beziehung zwischen Aktien erfassen können


# 12
LSTM 同样被用于从时间序列数据中提取抽象信息和特征. 但是LSTM由于其复杂的结构, 使得模型能够同时获取时间序列数据的长期和短期的信息。Das BiLSTM, bestehend aus zwei LSTM-Modellen mit entgegengesetzten Richtungen, kann die Informationen in beide Richtungen voll ausschöpfen.

LSTM 模型的应用场景和其他基于预测的深度学习模型以及机器学习模型都不同, 尽管LSTM也可以提取特征, 但是这些特征不再被用于预测, 而是用作深度强化学习的Entscheidungsfunktion的近似, 而这个决定函数的作用是将强化学习的状态空间映射到行为空间。因此LSTM模型的有效性不再和预测的准确性相关联。

但是, 由于深度学习自带的问题, 导致深度学习通常不会单独被应用于算法交易系统。首先, 过拟合问题无法被忽视。

Darüber hinaus können Methoden von DL die für algorithmischen Handel erforderliche kontinuierliche und schnelle Entscheidungsfindung nicht bewältigen. Daher werden Methoden von ML und DL häufig in Kombination mit RL eingesetzt.

# 13 
DRL als Kombination von DL und RL zeichnet sich durch seine hohe Leistung, starke Generalisierungsfähigkeit und hohe Anpassungsfähigkeit in der Entscheidungsfindung aus. RL wird zunehmend für algorithmischen Handel in verschiedenen Finanzmärkten angewandt.

Auf diese Folie ist ein hybrides Modell veranschaulicht.

Im Vergleich zu prognosebasierten Methoden ermöglicht das sog. DRL die Abbildung vom Zustandsraum zum Aktionsraum durch kontinuierliches und selbstgesteuertes Lernen. Die Festlegung einer spezifischen Handelsregel mithilfe DRL ist eine zentrale Forschungsfrage im Handel an Finanzmärkten.

In diesem Modell, DL dient als Merkmale-Abstrakter, um die Entscheidungsfunktion zu approximieren. RL kann danach dynamischen algorithmischen Handel optimieren, indem die Preis-Zeitreihe als Umgebung betrachtet wird. Agenten erlernen Strategien durch ein Verfahren von Versuch und Irrtum, wobei sie Aktionen ausführen und je nach Ergebnis positive oder negative Verstärkung erhalten. Eine Handelsleistungsfunktion U(θ), wie Gewinn, Nutzen oder risikoangepasste Rendite, wird verwendet, um die Parameter des Handelssystems direkt zu optimieren.

Daher wird RL eingesetzt, um die Gewichtungen in einem DL-Modell über den Gradientenanstieg anzupassen.

# 14
有的学者针对深度强化学习模型的环境和状态空间的模拟提出了优化。Ich werde zunächst die Methoden zur besseren Darstellung der Umwelt erklären.

Denn ein wichtiger Schritt bei der Entwicklung einer dynamischen algorithmischen Handelsstrategie durch RL ist die umfassende Darstellung der Umweltzustände. Um diesen Prozess zu gewährleisten, wird ein DCRL-Model entwickelt.

Das DCRL-Modell wird als Alternative zu herkömmlichen Zeitreihenanalysen zur Darstellung der Umweltzustände angesehen. Herkömmliche Methoden basieren meist auf festen Zeitintervallen, während das DCRL-Modell Preiszeitreihen in intrinsischen Zeitintervallen analysiert.

Das Modell erlernt dabei Zustände der Preiszeitreihe, um optimale dynamische Schwellenwerte für die DC-Ereignisanalyse zu identifizieren. Richtungsänderungen umfassen zwei Arten von Ereignissen: Aufwärtsbewegungen, die identifiziert werden, sobald die Preisänderung größer oder gleich einem festen Schwellenwert ist

2 Komponenten:

Der DCRL-basierte algorithmische Handel umfasst zwei Hauptkomponenten. Zunächst wird die DC-Ereignismethode mit einem dynamischen DC-Schwellenwert verwendet, um die Umweltzustände des Marktes zu identifizieren und darzustellen. Anschließend trifft der RL-Entscheidungsalgorithmus Entscheidungen und führt entsprechende Handelsaktionen aus.


# 15
 Um eine bessere Modellierung des Zustandsraums zu realisieren, können wir Sentimentanalyse in RL einbinden. Manche Grenze von den herkömmlichen Modellen sind gleichzeitig zu überwinden.

Wenn wir zukünftige Marktentwicklung ableiten möchten, können Merkmale auf Basis von Preisbewegungen als minimale Informationen zur Modellierung des Zustandsraums angesehen werden, insbesondere unter Einbeziehung der neuesten Preishistorie. Um die Grenze rein technischer algorithmische Handelsstrategien gleichzeitig zu überwinden, ist der Zustandsraum von Agenten ebenfalls mit Sentimentanalyse angereichert.

 Neue Merkmale werden dem traditionellen Zustandsspektrum hinzugefügt.

Zur Sicherstellung, dass alle Merkmale zur Modellleistung beitragen können, wurden alle Merkmale im Fenster auf eine einheitliche Skala normiert.







