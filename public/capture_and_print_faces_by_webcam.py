# pip install opencv-contrib-python

import cv2
#help(cv2.face) - para saber o que foi instalado

#pega o dispositivo da webcam
webcam = cv2.VideoCapture(0)

#importa o haarcascade para identificar rostos
harcascadeRosto = cv2.CascadeClassifier('resources/haarcascade/haarcascade_frontalface_default.xml')

#variáveis auxiliares

numFotos = 5
idAtual = 1
idPessoa = input("Digite o número da pessoa na sequência: ")
larguraImg, alturaImg = 220, 220

while True:
    #captura a conexão e imagem recebida da camera
    conectado, imagem = webcam.read()

    #converte a cor da imagem recebida para escala cinza
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    #busca rostos na imagem recebida
    rostosEncontrados = harcascadeRosto.detectMultiScale(imagemCinza, minNeighbors=8, minSize=(100,100))

    #print(rostosEncontrados)

    for (x, y, largura, altura) in rostosEncontrados:
        #desenha retangulo nos rostos encontradps
        cv2.rectangle(imagem, (x,y), (x+largura, y+altura), (0,0,255), 2)

        #salva a imagem na memória já com o tamanho modificado
        if cv2.waitKey(1) == ord('f'):
            imagemFace = cv2.resize(
                imagemCinza[y:y+altura, x:x+largura],
                (larguraImg, alturaImg) )

            #salva a imagem na pasta definida ex pessoa.1.10.jpg
            cv2.imwrite("resources/photos/pessoa."+str(idPessoa)
                        + "." + str(idAtual)+".jpg",
                        imagemFace)

            print("Imagem: "+str(idAtual)+ " capturada com sucesso!!!")
            idAtual+=1

    cv2.waitKey(3)
    cv2.imshow("Identifica", imagem)

    #finaliza o while apos as 225 fotos
    if(idAtual >= numFotos +1):
        break

print("Captura finalizada com sucesso!!!")

#libera a webcam
webcam.release()
#fecha a janela do cv2
cv2.destroyAllWindows()