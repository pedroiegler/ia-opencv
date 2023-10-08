import cv2
from cvzone.HandTrackingModule import HandDetector
import datetime

video = cv2.VideoCapture(0)

video.set(3, 1280)
video.set(4, 720)


detector = HandDetector(detectionCon=0.8)
estadoAtual = [0, 0, 0, 0, 0]
estado_senha = [[0, 1, 1, 0, 0],[0, 1, 0, 0, 0],[0, 1, 1, 0, 0]]
teste_senha = []
i=0
while True:
    _, img = video.read()
    hands, img = detector.findHands(img)

    if hands:
        estado = detector.fingersUp(hands[0])
        # print(estado)

        if estado != estadoAtual and estado != [0, 0, 0, 0, 0]:
            estadoAtual = estado
            teste_senha.append(estado)
            i+=1

        estadoAtual = estado
        if teste_senha != estado_senha and i==3:
            i=0
            print(teste_senha)
            teste_senha=[]

        elif teste_senha==estado_senha and i==3:
            print('finalizado')
            break
    cv2.imshow('img', cv2.resize(img, (640, 420)))
    cv2.waitKey(1)