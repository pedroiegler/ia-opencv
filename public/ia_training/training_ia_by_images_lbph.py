# Arquivo responsável por realizar a leitura das imagens e
# Gerar o arquivo .yml de reconhecimento das faces

#adiciona o OpenCv no projeto
import cv2

#adiciona a biblioteca do sistema
import os

#adiciona o NumPy ao projeto
import numpy as np

#carrega o algoritmo do LBPH na variavel para utilizarmos no código
lbph = cv2.face.LBPHFaceRecognizer_create()

#cria uma função que pega as imagens na pasta e
#retorna o id de identificação da pessoa na imagem
def pegaImagemId():
    #pega o caminho atual das imagens
    caminhos = [os.path.join('resources/photos', f) for f in os.listdir('resources/photos')]

    #print(caminhos)

    #cria um vetor para salvar as fotos dos rostos
    faces = []

    #cria um vetor para salvar o id da pessoa
    ids = []

    #passa por todos os caminhos das imagens
    for caminhoImagem in caminhos:
        #carrega a imagem da face e converte em cinza
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        #pega o id com base no nome da imagem
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        #adiciona o id atual ao vetor
        ids.append(id)

        #adiciona a imagem atual ao vetor
        faces.append(imagemFace)

        #mostra a imagem na tela
        cv2.imshow("face", imagemFace)
        cv2.waitKey(20)

    #fecha a janela e tira da memória
    cv2.destroyAllWindows()

    #retorna os ids e faces em forma de vetor
    return np.array(ids), faces

ids, faces = pegaImagemId()

print("Treinando ia com imagens.......")

#realizando o treinamento
lbph.train(faces, ids)

#salva o arquivo com o treinamento realizado
lbph.write('public/ia_training/treinamentoLBPH.yml')

print("Treinamento Realizado!!!")