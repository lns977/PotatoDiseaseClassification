import numpy as np
import streamlit as st
import cv2
import pickle
import tensorflow as tf

# Loading the model
model = pickle.load(open("D:/Potato Disease Classification/model/model_1.pkl",'rb'))

# Name of classes 
class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Setting the Title of the  App
st.title("Plant Leaf Disease Classification")
st.markdown("Upload an image of the plant's leaf")

# Uploading the image
plant_image = st.file_uploader("Choose an Image...", type=["jpg","png"])
submit = st.button("Predict")

#on predict button click
if submit:
    if plant_image is not None:
        
        #convert the file to an opencv image
        file_bytes = np.asarray(bytearray(plant_image.read()),dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        #Displaying the image
        st.image(opencv_image,channels="BGR")
        st.write(f"Image shape : {opencv_image.shape}")
        # Resizing the image
        opencv_image = cv2.resize(opencv_image,(256,256))

        #convert image to 4 dim
        opencv_image.shape = (1,256,256,3)

        #make prediction 
        y_pred = model.predict(opencv_image)
        result =  class_name[np.argmax(y_pred)]
        confidence = round(np.max(y_pred)*100,2)
        # Displaying the Result
        #st.title(str("This is "+result.split('-')[0]+"leaf with "+result.split('-')[1]))
        st.title(result)
        st.title(confidence)
        