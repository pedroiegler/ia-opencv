import cv2
import numpy as np
import os

haarcascadeRosto = cv2.CascadeClassifier('resources/haarcascade/haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

lbph = cv2.face.LBPHFaceRecognizer_create()

lbph.read('public/ia_training/treinamentoLBPH.yml')

largura, altura = 220, 220

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

nomes = ['Pessoa1', 'Pessoa2', 'Pessoa3']

while(True):
    
    conectado, imagem = webcam.read()
    
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    facesEncontrada = haarcascadeRosto.detectMultiScale(imagemCinza)
    
    for(x, y, a, l) in facesEncontrada:
        
        imagemFace = cv2.resize(imagemCinza[y:y+a, x:x+l], (largura, altura))
        
        cv2.rectangle(imagem, (x,y), (x+l, y+a), (0,0,255), 2)
        
        id, correspondencia = lbph.predict(imagemFace)
        
        if(correspondencia < 50):
            cv2.putText(imagem, nomes[id-1], (x, y+a+30), font, 2, (0,0,255))
        
    cv2.imshow('face', imagem)
    
    if(cv2.waitKey(1) == ord('q')):
        break
    
webcam.release()
cv2.destroyAllWindows()