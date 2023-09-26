import cv2

rostoFrontal = cv2.CascadeClassifier('resources/haarcascade/cascade.xml')

img = cv2.imread('resources/images/images.png')

imagemCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rostosEncontrados = rostoFrontal.detectMultiScale(imagemCinza)

print(rostosEncontrados)

for (x, y, altura, largura) in rostosEncontrados:
    cv2.rectangle(img, (x,y), (x + altura, y + largura), (0,255,0), 4)
    

cv2.imshow("window", img)

cv2.waitKey()