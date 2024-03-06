# -*- coding: utf-8 -*-
"""
Created on Monday Jan 9 11:59:34 2023

@author: morte
"""

import numpy as np # linear algebra
from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
import os


folder_path='C:/Users/Morteza/Desktop/YouTube/coding/CNN_Tutorial_color/input/' 
images1 = []
for img in os.listdir(folder_path):
    img=folder_path+img
    img = load_img(img, target_size=(20,20)) 
    img = img_to_array(img)/ 255    
    X=img
    images1.append(X)

folder_path='C:/Users/Morteza/Desktop/YouTube/coding/CNN_Tutorial_color/output/' 
images2 = []
for img in os.listdir(folder_path):
 
    img=folder_path+img
    img = load_img(img, target_size=(20,20)) 
    img = img_to_array(img)/ 255
    Y=img
    images2.append(Y)
    
X = np.array(images1)
Y = np.array(images2)

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 3)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3,3), activation='sigmoid', padding='same'))

model.compile(optimizer='rmsprop', loss='mse')
model.fit(x=X, y=Y, batch_size=80, epochs=500, verbose=1)

model.evaluate(X, Y, batch_size=80)


folder_path='C:/Users/Morteza/Desktop/YouTube/coding/CNN_Tutorial_color/Testing_input/' 
img='imagein1.png'
img=folder_path+img
img = load_img(img, target_size=(20,20)) 
img = img_to_array(img)/ 255

X = np.array(img)
X = np.expand_dims(X, axis=2)
X=np.reshape(X,(1,20,20,3))

output = model.predict(X)

output=np.reshape(output,(20,20,3))

import matplotlib.pyplot as plt
imshow(output)
plt.show()