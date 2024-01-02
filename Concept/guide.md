# Konzept Guide
Grundlage ist für Anwendung des Konzepts ist die Instalaltion von Python 3.4. 
Das Konzept basiert auf der Grundlage der KI gestützten Objekterkennung am Beispiel von Obst und Gemüse. Dabei orientiert sich das Konzept an einer Quantität von rund 500 zu klassifizierenden Produkten.
Mit den hier nidergelegten Schritten, soll die Umsetzung des Konzeptes realisiert werden können. Anpassungen an individuelle Bedürfnisse, technosche Vorraussetzungen oder andere Grundlagen werden nicht bedacht und sind entsprechend bei bedarf oder zu weiteren Optimierung individuel anzupassen.

## Inhaltsverzeichnis

1. [Datenerhebung](#datenerhebung)
    * 1.1. Aufbau des Systems
    * 1.2. Erhebung der Daten
2. [Training](#training)
3. [Anwendung](#anwendung)


## Datenerhebung
Benötigte Utensilien:
- POS-System (Kasse, SB-Waage, SCO)
- Kamera (Full-HD)
- Computer
- ggf. Stativarm

### Aufbau des Systems
1. Auswahl des verwendeten POS-Systems
2. Befästigung der Kamera
    Die ausgewählte Full-HD fähige Kamera ist in einer höhe von 20cm zur Grundfläche des POS-Systems anzubringen. Dabei ist auf einem 45 Grad-Winkel dieser zur Grundfläche zu achten. Außerdem muss der Bildausschnitt den Gesammten Erfassungsbereich in gänze abdecken.
    Zur Umsetzung einer dauerhaften Befästigung können je nach System zusätzliche Hilfsmittel wie Stativarme notwendig sein.
3. Aufbau des Computer-Setups und anschließen der eingerichteten Kamera

### Erhebung der Daten
1. Terminal öffnen
2. Libarys installieren
```
pip install opencv-python
pip install os_sys 
```
3. öffne [data_generation.py](Concept\data_generation.py)  
Vor dem Auführen sicherstellen das die angegebene Kamera an das System angeschlossen ist. 
4. Sicherstellen das Korekte Kamera und Dateipfad angegeben ist.
```
dir = "dataset/product_name/"
cap = cv2.VideoCapture(0) 
```
*Bei Falsch ausgewählter Kamera ist über die Zahl in der Funktion zu iterieren.

5. Automatisiertes Aufnehmen  
Um den Ablauf der Datenerhebung nicht für jede Aufhnahme erneut zu starten, kann mittels schleifenartiger Wiederholung der Funktion createTrainingset(), automatisert werden. 

# Training
1. Terminal öffnen
2. Libarys installieren
```
pip install numpy
pip install —upgrade keras
```
3. Anwendungsparameter definieren
```
dataset_path = '/dataset'
num_classes = 500
training_epochs = 10
# Definieren der erfassten Bildgröße
image_size = (1080 , 1920)
```
Zu definieren ist der oben verwendete Datensatzordner, Anzahl an Klassen, Trainingsepocen und die Angabe der Bildgröße der in der erhobenen Bilder.
Anpassungen abweichend vom der hier und in der zugrundeliegenden Bachelorarbeit thematisierten Konzept, sind möglich. 

4. [model_train.py](Concept\model_train.py) Ausführen und Model erhalten
5. weitere optionale Anpassungen
Anpassung der Automatisierten Datenanpassung und Datennormalisierung durch Änderung der Parameter.
```
data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=20,
    height_shift_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip= True,
    validation_split=0.2
)
```
Änderung der Batch größe des Batchlernverfahrens
```
batch_size=32, 
```
Anpassung der Cpnvolutional und Pooling Schicht
*Angabe hier Anzahl an Neuronen und größe der verwendeten Filtermatrx, sowie der Aktivierungsfunktion. Zudem eine Pooling Schicht mit Angabe der Pooling-Matrix.
```
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
```
Anpassung der Dense Schicht 
*Angabe hier Anzahl an Neuronen und verwendete Aktivierungsfunktion
```
model.add(Dense(1024, activation='relu'))
```
Je nach Bedarf können weitere Schichten hinzugefügt oder weggelassen werden. 
Zur konkreten Anpassung ist die Dokumentation von [Keras](https://keras.io/getting_started/)   durchzulesen. 

# Anwendung
1. Terminal Öffnen
2. Libary downloaden
```
pip install Pillow
```
3. Angabe aller Klassennamen der verwendeten Objekte.  
Angabe wie in der Ordnerstruktur. (meist Alphabetisch)
```
classes = [
    "klasse-1",
    "klasse-2",
    # ....
    "klasse-n"
]
```

4. Sicherstellen, dass das Model und die Kamera korekt hinterlegt sind.
```
model = load_model('og_model.h5')
cap = cv2.VideoCapture(0)
```

5. [model_execution.py](Concept\model_execution.py) Ausführen