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

option_colour = st.sidebar.selectbox('Colour model', ('HSV', 'RGB'))

cola, colb, colc, cold = st.sidebar.columns(4)
with cola: height_1 = st.text_input('Height 1', '800')
with colb: width_1 = st.text_input('Width 1', '1400')
with colc: height_2 = st.text_input('Height 2', '1900')
with cold: width_2 = st.text_input('Width 2', '2500')

tab1, tab2 = st.tabs(["Analysis", "Image"])

if img_file_buffer is not None:
    
    image = Image.open(img_file_buffer)
    img_array = np.array(image) # if you want to pass it to OpenCV
    img = img_array[int(height_1):int(height_2),int(width_1):int(width_2),:]
    plaqueta = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with tab1:    
    
        if option_colour == 'HSV' :
    
            img_hsv = cv2.cvtColor(plaqueta, cv2.COLOR_RGB2HSV)
            h,s,v = cv2.split(img_hsv)
    
            with st.container():
    
                col1, col2, col3 = st.columns(3)
    
                with col1:
                    st.image(h, caption='H', width = 220)
                    valueH = st.slider('H', 0, 255, (0, 255))
                with col2:
                    st.image(s, caption='S', width = 220)
                    valueS = st.slider('S', 0, 255, (0, 255))
                with col3:
                    st.image(v, caption='V', width = 220)
                    valueV = st.slider('V', 0, 255, (0, 255))
    
    
            with st.container():
    
                col11, col22, col33 = st.columns(3)
    
                with col11:
                    st.image(plaqueta, caption='Plaqueta', width = 220)
                with col22:
                    lower_hsv = np.array([valueH[0],valueS[0],valueV[0]], dtype=np.uint8)
                    upper_hsv = np.array([valueH[1],valueS[1],valueV[1]], dtype=np.uint8)
                    mask_sucio_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
                    st.image(mask_sucio_hsv, caption='MASK', width = 220)
                with col33:
                    pix_im = (int(height_2)-int(height_1))*(int(width_2)-int(width_1))
                    pix_mask = int(np.sum(mask_sucio_hsv/255))
                    # 1 píxel - 0.00311 cm
                    area_tot_mm2 = ((int(height_2)-int(height_1))*0.00311)*((int(width_2)-int(width_1))*0.00311)
                    st.sidebar.header('Área total: ' + "{:.2f}".format(area_tot_mm2) + ' cm2')
                    st.sidebar.header('Sucio plaqueta: ' + "{:.2f}".format(pix_mask/pix_im*100) + ' %')
    
        if option_colour == 'RGB' :
    
            r,g,b = cv2.split(plaqueta)
    
            with st.container():
    
                col1, col2, col3 = st.columns(3)
    
                with col1:
                    st.image(r, caption='R', width = 220)
                    valueR = st.slider('R', 0, 255, (0, 255))
                with col2:
                    st.image(g, caption='G', width = 220)
                    valueG = st.slider('G', 0, 255, (0, 255))
                with col3:
                    st.image(b, caption='B', width = 220)
                    valueB = st.slider('B', 0, 255, (0, 255))
    
    
            with st.container():
    
                col11, col22, col33 = st.columns(3)
    
                with col11:
                    st.image(plaqueta, caption='Plaqueta', width = 220)
                with col22:
                    lower = np.array([valueR[0],valueG[0],valueB[0]], dtype=np.uint8)
                    upper = np.array([valueR[1],valueG[1],valueB[1]], dtype=np.uint8)
                    mask_sucio = cv2.inRange(plaqueta, lower, upper)
                    st.image(mask_sucio, caption='MASK', width = 220)
                with col33:
                    pix_im = (int(height_2)-int(height_1))*(int(width_2)-int(width_1))
                    pix_mask = np.int(np.sum(mask_sucio/255))
                    # 1 píxel - 0.00311 cm
                    area_tot_mm2 = ((int(height_2)-int(height_1))*0.00311)*((int(width_2)-int(width_1))*0.00311)
                    st.sidebar.header('Área total: ' + "{:.2f}".format(area_tot_mm2) + ' cm2')
                    st.sidebar.header('Sucio plaqueta: ' + "{:.2f}".format(pix_mask/pix_im*100) + ' %')
    
    with tab2:   
    
        # Cargar la imagen
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
        fig = px.imshow(img_rgb)
            
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(
            dragmode = "zoom",
            autosize=False,
            width=1500,
            height=800,)
        
        st.plotly_chart(fig, True)
