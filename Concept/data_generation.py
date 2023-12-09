import cv2
import os 

dir = "dataset/product_name/"

def doFoto(dirname):
    cap = cv2.VideoCapture(0)
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

def createFileName(dirname):
    i = 0
    for osl in os.listdir(dirname):
        i = i+1
    fileName = dirname + str(i+1) + ".jpg"
    return fileName  

def createTrainingset():
    dirname = createFileName(dir)
    doFoto(dirname)