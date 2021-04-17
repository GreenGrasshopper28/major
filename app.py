import streamlit as st
from tensorflow import keras
from keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from PIL import Image
import cv2
from keras.applications.mobilenet_v2 import preprocess_input,decode_predictions


st.title("Image classifier")
upload=st.file_uploader('Label = Upload the image')
model=MobileNetV2()
file_bytes=np.asarray(bytearray(upload.read()),dtype=np.uint8)
opencv_image=cv2.imdecode(file_bytes,1)
opencv_image=cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
img = Image.open(upload)
st.image(img,caption='Uploaded Image',width=300)
model = MobileNetV2()
x = cv2.resize(opencv_image,(224,224))
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
y=model.predict(x)
  #label=decode_predictions(y)
#st.title(label)
if st.button('PREDICT'):
  st.write('Result...')
  label=decode_predictions(y)
  st.title(label)