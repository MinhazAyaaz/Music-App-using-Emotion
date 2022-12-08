import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

new_model = tf.keras.models.load_model('CSE499A_Model.h5')

frame = cv2.imread("happyboy.jpg")

frame.shape

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#importing the face detection algorithm

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

gray.shape

faces = faceCascade.detectMultiScale(gray,1.2,4)
for x,y,w,h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x=x+w]
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Face not detected")
    else:
        for(ex,ey,ew,eh) in facess:
            face_roi = roi_color[ey: ey+eh, ex: ex+ew]

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

final_image = cv2.resize(face_roi, (224,224))
final_image = np.expnd_dims(final_image,axis=0) ##adding fourting dimetion
final_image = final_image / 255.0 ## normalizing

Predictions = new_model.predict(final_image) ##using the model to analyze the picture
Predictions[0] ## showing the prediction array
np.argmax(Predictions) ##finding dominant emotion
