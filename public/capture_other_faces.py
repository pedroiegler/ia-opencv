import cv2

rostoFrontal = cv2.CascadeClassifier('resources/haarcascade/haarcascade_frontalface_default.xml')
rostoGato = cv2.CascadeClassifier('resources/haarcascade/haarcascade_frontalcatface.xml')

img = cv2.imread('resources/images/cat.jpg')

imagemCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rostosEncontrados = rostoFrontal.detectMultiScale(imagemCinza)
rostosGato = rostoGato.detectMultiScale(imagemCinza)

print(rostosEncontrados)
print(rostosGato)

for (x, y, altura, largura) in rostosEncontrados:
    cv2.rectangle(img, (x,y), (x + altura, y + largura), (0,255,0), 4)
    

for (x, y, altura, largura) in rostosGato:
    cv2.rectangle(img, (x,y), (x + altura, y + largura), (0,255,0), 4)
    

cv2.imshow("window", img)

cv2.waitKey()