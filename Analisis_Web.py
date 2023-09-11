import streamlit as st
import os
from os import path
from os import walk
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

img_file_buffer = st.file_uploader("Upload an image")
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image) # if you want to pass it to OpenCV
    st.image(image, caption="The caption", use_column_width=True)


            

