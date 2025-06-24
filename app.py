#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np

model = YOLO('runs/detect/train/weights/best.pt')  

st.title("Fruit Detection Demo using YOLOv8")
st.write("Upload an image or video to detect objects.")

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type.startswith("image"):
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", use_container_width=True)

        img_array = np.array(image)
        results = model(img_array)

        res_plotted = results[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption="Predicted Output", use_container_width=True)

    
    elif file_type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.video(tfile.name)
        st.info(" Video inference demo not shown frame-by-frame here. You can save predictions using model.predict(save=True) in a separate script.")

