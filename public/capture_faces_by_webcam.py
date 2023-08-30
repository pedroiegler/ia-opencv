import cv2

haarRosto = cv2.CascadeClassifier("resources/haarcascade/haarcascade_frontalface_default.xml")
webcam = cv2.VideoCapture(0)  

while True:
    conectado, imagem = webcam.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    rostosEncontrados = haarRosto.detectMultiScale(imagemCinza,
                                                   scaleFactor=1.1,
                                                   minNeighbors=3,
                                                   minSize=(50,50),
                                                   maxSize=(200,200)
                                                   )
    
    for(x, y, largura, altura) in rostosEncontrados:
        cv2.rectangle(imagem, (x,y), (x+largura,y+altura), (0,0,255), 4)
    
    cv2.waitKey(1)
    cv2.imshow("webcam", imagem)
    
    if (cv2.waitKey(1) == ord('x')):
        breakxx


webcam.release()
cv2.destroyAllWindows()
