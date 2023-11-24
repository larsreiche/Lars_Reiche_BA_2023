import requests
import cv2
import numpy as np
import os 

urls = ["http://192.168.178.20:4747/video", "http://192.168.178.29:4747/video", "http://192.168.178.30:4747/video" ]
dirs = ['kampos0/zwetschgen/','kampos1/zwetschgen/', 'kampos2/zwetschgen/']

def doFoto(url,dirname):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Die Kamera konnte nicht geöffnet werden.")
    else:
        print("Kamera geöffnet")
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(dirname, frame)
            print("Foto wurde aufgenommen und gespeichert.")
        cap.release()
    cv2.destroyAllWindows()

def createFileName(path):
    i = 0
    for osl in os.listdir(path):
        i = i+1
    fileName = path + str(i+1) + ".jpg"
    return fileName   

def createTrainingset():
    i = 0
    for url in urls:
        dirname = createFileName(dirs[i])
        doFoto(url,dirname)
        i = i+1 

