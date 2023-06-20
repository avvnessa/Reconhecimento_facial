import cv2
import os
import imutils

pessoaNome = 'Fabio'
dataPath = 'C:/Users/FF-admin/Videos/Reconhecimento Facial em video/Data'
pessoaPath = dataPath + '/' + pessoaNome

if not os.path.exists(pessoaPath):
	print('Pasta Criada:', pessoaPath)
	os.makedirs(pessoaPath)

cap = cv2.VideoCapture('fabio2.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0
	
while True:
    
    ret, frame = cap.read()
    if ret == False: break
    frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rosto = auxFrame[y:y+h,x:x+w]
        rosto = cv2.resize(rosto,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(pessoaPath + '/rosto_{}.jpg'.format(count),rosto)
        count = count + 1
        cv2.imshow('frame',frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 500:
    	break

cap.release()
cv2.destroyAllWindows()