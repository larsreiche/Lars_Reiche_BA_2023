# Bachelorarbeit KI gestütze Objekterkennung
Der hier niedergelegte Code ist Teil der Bachelorarbeit zum Thema "Optimierung des Kassiervorgangs im Lebensmitteleinzelhandel
anhand von KI-gestützter Objekterkennung von Obst und Gemüse" von Lars Reiche.
Inhalt dieser ist sowohl der Untersuchung zugrundeliegende Code und erhobenen Daten, als auch das daraus resultierende Konzept.


## Inhaltsverzeichnis

1. [Untersuchung](#untersuchung)
    * 1.1. [data_generation.py](BA_investigation\data_generation.py)
    * 1.2. [datasets](BA_investigation\datasets)
    * 1.3. [model_train.py](BA_investigation\model_train.py)
    * 1.4. [model_evaluation.py](BA_investigation\model_evaluation.py)

2. [Konzept](#konzept)
    * 1.4. [data_generation.py](Concept\data_generation.py)
    * 1.4. [model_train.py](Concept\model_train.py)
    * 1.4. [model_execution.py](Concept\model_execution.py)
    * 1.4. [Konzept Guide](Concept\guide.md)
3. Schluss

# Untersuchung
Die Untersuchung befasste sich mit der Frage welchen Einfluss die Kameraposition auf die KI gestützte Objekterkennung am POS hat. Dazu wurden drei verschieden Kamerapositionen gegeneinander ealuiert. Diese waren frontal auf Produkthöhe, in einem 45 Grad Winkel und in einem 90 Grad Winkel direkt von oben über den Produkten.  
Im Untersuchungsrahmen wurden 10 Produkte aus den drei besagten Kamerapositionen betrachtet. Trainiert wurde dazu eine Convolutional Neural Network auf basis von Keras mit zwei Convolutional Schichten mit jeweils einer darauffolgenden pooling Schicht. Insgeammt wurden in diesem 2048 Merkmale ausgeprägt, die in dem darauffolgenden Dense Schichten mit 64+10 Neuronen weiterverarbeitet wurden.
Basierend auf den Erkenntnissen dieser Untersuchung wurde das nachstehende Konzept der Arbeit entwickelt.
Der Konkrete Erkenntnisgewinn ist in der Arbeit ausreichend beschriebe, sodass dies nicht weiter behandelt wird und entsprechend nachgelesen werden kann. 

# Konzept
Grundlage des Konzepts sind die gewonnen Erkenntnisse der vorrangegangenen Untersuchung. 
Dabei orientiert sich das Konzept an einer Quantität von rund 500 zu klassifizierenden Produkten, um das oben beschriebene Problem in entsprechendem praxisnahen Ausmaß optimieren und umsetzen zu können. 
Konkretere Handlungschritte zur Umsetzung des Konzepts sind im entsprechenden [Konzept Guide](Concept\guide.md) nachzulesen.
Anpassungen an individuelle Bedürfnisse, technosche Vorraussetzungen oder andere Grundlagen werden nicht bedacht und sind entsprechend bei bedarf oder zu weiteren Optimierung individuel anzupassen.
Das Konzept bietet somit die Grundlage zur Optimierung des Kassiervorgangs im Lebensmitteleinzelhandel anhand von KI-gestützter Objekterkennung von Obst und Gemüse am POS.