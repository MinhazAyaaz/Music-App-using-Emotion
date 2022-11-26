import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

## In this section we are loading the data and normalizing it

Datadirectory = "train/"
os.chdir('E:\CSE499A')

Classes = ["0","1","2","3","4","5","6"]

img_size = 224

training_Data = []

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory,category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass
        
create_training_Data()  

## Test print to check if all data are loaded
print(len(training_Data))  

training_Data.shape() 

random.shuffle(training_Data)

X = []
Y = []

for features,label in training_Data:
    X.append(features)
    Y.append(label)
    
X = np.array(X).reshape(-1,img_size,img_size,3)
    
X = X/255.0

## In this section we are deep learning model for training using Transfer Training

model = tf.keras.applications.MobileNetV2() ## Pre-trained Model

model.summary()

base_input = model.layers[0].input ##input

base_output = model.layers[-2].output

base_output ##checking

final_output = layers.Dense(128)(base_output) ##adding new layer, after the output of global pooling layer
final_output = layers.Activation('relu')(final_output) ## Activation function
final_output = layers.Dense(64)(final_output) 
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7,activation='softmax')(final_output) ## my classes are 07, classification layer

final_output ##checking

new_model = keras.Model(inputs = base_input, outputs = final_output)

new_model.summary()

new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

Y = np.array(Y)

new_model.fit(X,Y, epochs = 15) ##Change to 25 if accuracy low

new_model.save("CSE499A_Model.h5")



































