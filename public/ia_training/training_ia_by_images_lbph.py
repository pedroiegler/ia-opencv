import cv2
import numpy as np
import os

lbph = cv2.face.LBPHFaceRecognizer_create()

def getImgId():
    caminhos = [os.path.join('photos', f) for f in os.listdir('photos')]
    
    #print(caminhos)
    
    faces = []
    ids = []
    
    for caminhoImagem in caminhos:
        imageFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        
        ids.append(id)
        faces.append(imageFace)
        
        cv2.imshow("face", imageFace)
        cv2.waitKey(150)
    
    cv2.destroyAllWindows()
    
    return np.array(ids), faces
    
ids, faces = getImgId()

print("TREINANDO IA COM IMAGENS...")
lbph.train(faces, ids)
lbph.write("public/ia_training/treinamentoLBPH.yml")
print("TREINAMENTO REALIZADO!")