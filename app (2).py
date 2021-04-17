import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from PIL import Image # Strreamlit works with PIL library very easily for Images
import cv2
from keras.applications.mobilenet_v2 import preprocess_input,decode_predictions
st.title("Image-Classifier: Cats and Dogs")
upload = st.sidebar.file_uploader(label='Upload the Image')
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
  model = MobileNetV2()
  x = cv2.resize(opencv_image,(224,224))
  x = np.expand_dims(x,axis=0)
  x = preprocess_input(x)
  y = model.predict(x)
  if st.sidebar.button('PREDICT'):
    st.sidebar.write('Result...')
    label = decode_predictions(y)
    for i in range(3):
      out = label[0][i]
      st.sidebar.title('%s (%.2f%%)' % (out[1], out[2]*100))