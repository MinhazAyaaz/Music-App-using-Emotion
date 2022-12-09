import keras
import cv2
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,load_img
datagen = ImageDataGenerator(rotation_range =15, 
                         width_shift_range = 0.2, 
                         height_shift_range = 0.2,  
                         rescale=1./255, 
                         shear_range=0.2, 
                         zoom_range=0.2, 
                         horizontal_flip = True, 
                         fill_mode = 'nearest', 
                         data_format='channels_last', 
                         brightness_range=[0.5, 1.5]) 


img_dir = "E:\Music-App-using-Emotion\ImageData" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)

    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    path, dirs, files = next(os.walk("E:\Music-App-using-Emotion\ImageData"))
    file_count = len(files) #to find number of files in folder

    for batch in datagen.flow (x, batch_size=1, save_to_dir =r'E:\Music-App-using-Emotion\trainNew\resize',save_prefix="a",save_format='jpg'):
        i+=1
        if i==file_count:
            break