import os
import sys
import cv2
import glob
import torch
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from utils import load_model, inference
import warnings
warnings.filterwarnings("ignore")

model = load_model("./assets/model", "model_ckpt.pth")

# Streamlit application main function
def main():
    st.title("Detect Marching Band Members!")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        
        # add bounding boxes to the image
        box_image = inference(model, image, score_thresh=0.75)
        st.image(box_image, caption='Box Image', use_column_width=True)

if __name__ == "__main__":
    main()
