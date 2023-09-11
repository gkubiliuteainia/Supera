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

img_file_buffer = st.sidebar.file_uploader("Upload an image")
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image) # if you want to pass it to OpenCV
    st.image(image, caption="The caption", use_column_width=True)

option_colour = st.sidebar.selectbox('Colour model', ('HSV', 'RGB'))

cola, colb, colc, cold = st.sidebar.columns(4)
with cola: height_1 = st.text_input('Height 1', '800')
with colb: width_1 = st.text_input('Width 1', '1400')
with colc: height_2 = st.text_input('Height 2', '1900')
with cold: width_2 = st.text_input('Width 2', '2500')

img_raw = cv2.fromarray(img_array)
img = img_raw[int(height_1):int(height_2),int(width_1):int(width_2),:]
plaqueta = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
