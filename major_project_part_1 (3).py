# -*- coding: utf-8 -*-
"""major project part 1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10g-TzGZGaVAz6GRL-u1m28PdYkX_V6V0
"""

!pip install --upgrade pip

!pip install tensorflow-cpu

from tensorflow import keras
from keras.applications.mobilenet_v2 import MobileNetV2

# Input images will be in (224, 224, 3)
model = MobileNetV2(include_top=False)
model.summary()

for layer in model.layers:
  layer.trainable=False

!pip install bing-image-downloader

import cv2
cv2.__version__

!pip install opencv-python==3.4.13.47 --quiet
!pip install cvlib --quiet

!mkdir images

from bing_image_downloader import downloader
downloader.download("dogs",limit=15,output_dir='images',adult_filter_off=True)
downloader.download("cats",limit=15,output_dir='images',adult_filter_off=True)

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow import keras
from keras.applications.mobilenet_v2 import MobileNetV2
import matplotlib.pyplot as plt
from keras.preprocessing import image

MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    
)

target=[]
images=[]
flat_data=[]

DATADIR = '/content/images'
CATEGORIES = ['dogs','cats']

for category in CATEGORIES:
  class_num=CATEGORIES.index(category)
  path=os.path.join(DATADIR,category)
  #print(path)
  for img in os.listdir(path):
    img_array=imread(os.path.join(path,img))
    #print(img_array)
    #plt.imshow(img_array)
    img_resized=resize(img_array,(150,150,3))
    flat_data.append(img_resized.flatten())
    images.append(img_resized)
    target.append(class_num)
flat_data=np.array(flat_data)
target=np.array(target)
images=np.array(images)

flat_data[0]

target

unique,count=np.unique(target,return_counts=True)
plt.bar(CATEGORIES,count)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(flat_data,target,test_size=0.3,random_state=109)

from sklearn.model_selection import GridSearchCV
from sklearn import svm
param_grid=[
            {'C':[1,10,100,1000],'kernel':['linear']},
             {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']},
            ] 
            
svc=svm.SVC(probability=True)
clf=GridSearchCV(svc,param_grid)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
y_pred

y_test

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score(y_pred,y_test)

confusion_matrix(y_pred,y_test)

from keras.applications.mobilenet_v2 import preprocess_input,decode_predictions

#!pip install h5py

model.save('major project part 1.hdf5')



