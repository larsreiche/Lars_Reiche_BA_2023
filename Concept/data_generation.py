import cv2
import os 

# Angabe des Speicherorts der zusammelnden Bilder einer Klasse
dir = "dataset/product_name/"
# Angabe d. verwendeten Systemkamera
cap = cv2.VideoCapture(0) 

# Aufnahme eines Einzelfotos
def doFoto(dirname):
    if not cap.isOpened():
        print("Die Kamera konnte nicht geöffnet werden.")
    else:
        print("Kamera geöffnet")
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(dirname, frame)
            print("Foto "+dirname+" wurde aufgenommen und gespeichert.")
        cap.release()
    cv2.destroyAllWindows()

# Erstellt passenden Dateinamen
def createFileName(dirname):
    i = 0
    for osl in os.listdir(dirname):
        i = i+1
    fileName = dirname + str(i+1) + ".jpg"
    return fileName  

def createTrainingset():
    dirname = createFileName(dir)
    doFoto(dirname)

# createTrainingset()