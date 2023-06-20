import cv2
import os
import numpy as np

dataPath = 'C:/Users/FF-admin/Videos/Reconhecimento Facial em video/Data'
peopleList = os.listdir(dataPath)
print('Lista de pessoas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Lendo as imagens')

    for fileName in os.listdir(personPath):
            print('Rostos: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            #image = cv2.imread(personPath+'/'+fileName,0)
            #cv2.imshow('image',image)
            #cv2.waitKey(10)
    label = label + 1

#print('labels= ',labels)
#print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Treiando o reconhecedor de rostos
print("Treinando...")
face_recognizer.train(facesData, np.array(labels))

#Salvando o modelo obtido
face_recognizer.write('modeloLBPHFace.xml')
print("Salvando Modelo...")
