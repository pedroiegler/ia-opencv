import cv2
#help(cv2.face)

webcam = cv2.VideoCapture(0)

haarcascadeRosto = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

numFotos = 5
idAtual = 1
idPessoa = input("Digite o número da pessoa da sequencia: ")
larguraImg, alturaImg = 220, 220


while True:
    conectado, imagem = webcam.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    rostosEncontrados = haarcascadeRosto.detectMultiScale(imagemCinza, minNeighbors=8)
    
    print(rostosEncontrados)
    
    for(x, y, largura, altura) in rostosEncontrados:
        cv2.rectangle(imagem, (x,y), (x+largura, y+altura), (0,0,255), 2)
        
        if cv2.waitKey(1) == ord('f'):
            imageFace = cv2.resize(
                imagemCinza[y:y+altura, x:x+largura],
                (larguraImg, alturaImg))
            
            cv2.imwrite("photos/pessoa."+str(idPessoa) + "." + str(idAtual) + ".jpg", imageFace)
            
            print("Imagem: " + str(idAtual) + " Capturada com sucesso!")
            idAtual+=1
        
    cv2.waitKey(3)
    
    cv2.imshow("Identifica", imagem)
    
    if(idAtual >= numFotos+1):
        break
    
print("Captura finalizada com sucesso!")

webcam.release()
cv2.destroyAllWindows()