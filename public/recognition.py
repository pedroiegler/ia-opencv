#import o OpenCV para o projeto

import cv2

#help(cv2.face)

#pega o dispositvo da webcam
webcam = cv2.VideoCapture(0)

#importa o haarcascade para identificar rostos
haarcascadeRosto = cv2.CascadeClassifier('resources/haarcascade/haarcascade_frontalface_default.xml')

#atribui o LBPH na variavel para usar depois
lbph = cv2.face.LBPHFaceRecognizer_create()

#carrega o arquivo treinado no reconhecedor
lbph.read('public/ia_training/treinamentoLBPH.yml')

#define a altura e largura das imagens treinadas e capturadas
largura, altura = 220, 220

#carrega um arquivo de fonte para escrever na imagem
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

nomes = ['Pessoa1', 'Pessoa']


while (True):
    #Realiza a leitura da webcam
    conectado, imagem = webcam.read()

    #converte em escala de cinza
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGRA2GRAY)

    #encontra as faces na imagem
    facesEncontradas = haarcascadeRosto.detectMultiScale(imagemCinza)

    #passa por todos os rostos
    for(x, y, a, l) in facesEncontradas:
       #recorta apenas a imagem da face encontrada
       imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))

        #desenha o retangulo na imagem original
       cv2.rectangle(imagem, (x, y), (x+l, y+a), (0,0,255), 2)

       #pega o id da pessoa reconhecida e a porcentagem de correspondencia
       id, correspondencia = lbph.predict(imagemFace)
       print(id)
       #escreve em baixo do retangulo da imagem
       if correspondencia < 50:
        cv2.putText(imagem, nomes[id-1], (x, y+a+30), font, 2, (0,0,255))
    #exibe os resultados na tela
    cv2.imshow('Face', imagem)

    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()