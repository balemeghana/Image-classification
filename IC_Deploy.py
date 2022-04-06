# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 03:40:14 2022

@author: PRANAV
"""
import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2

st.title('Image Classification using CNN model')
upload = st.file_uploader('Label = Upload the image')
model_new = keras.models.load_model("D:\Python_Files\Image_Classification1.h5")
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()),dtype = np.uint8)
  opencv_image = cv2.imdecode(file_bytes,1)
  classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
  img = Image.open(upload)
  st.image(img,caption = 'Uploaded Image',width = 300)
 # opencv_image/=255.0
  x = cv2.resize(opencv_image,(180,180), interpolation=cv2.INTER_NEAREST)
  x = np.expand_dims(x,axis = 0)
  x = preprocess_input(x)
  y = model_new.predict(x)
  st.write('OUTPUT: ')
  st.write('Result: ',classes[np.argmax(y)])
