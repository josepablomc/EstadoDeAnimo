'''
PyPower Projects
Emotion Detection Using AI
'''

#USAGE : python test.py

from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Enojado','Feliz','Neutral','Triste','Sorprendido']

# Capturamos la imagen de la camara, el argumento requiere que pongamos el valor 0 para permite que se acceda a la camara.
cap = cv2.VideoCapture(0)



# While True continua la prediccion sin parar, en la linea 55 definimos la salida con la letra q en el teclado
while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
# Ppara la mejor lectura de los gestos pasamos la imagen a escala de grises 
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# Detecta margenes y guarda las imagenes en la variable faces
    faces = face_classifier.detectMultiScale(gray,1.3,5)

# Creamos las 4 coordenadas que se utilizan para ubicar los gestos de la cara
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
# Utiliza la prediccion basandose en lo mas cercano en las imagenes capturadas comparando con la camara. Las siguientes 4 lineas son las responsables de toda la prediccion.

            preds = classifier.predict(roi)[0]
            print("\nprediction = ",preds)
            label=class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Emotion Detector',frame)
# Definimos la salida con la letra q en el teclado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























