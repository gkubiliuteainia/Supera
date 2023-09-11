import streamlit as st
import time
import os
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from scipy import ndimage
from os import walk
from os import path

# ML
import joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


basepath = r'G:\SUPERA'

path_name = st.sidebar.text_input('Folder name', 'Robot')
folder_path = os.path.join(basepath, path_name)  # Construye la ruta usando os.path.join

archivos = []

for root, dirs, files in walk(basepath + '\\' + path_name):
    for file in files:
        if file.endswith(".bmp"):
            archivos.append(file)
        if file.endswith(".jpg"):
            archivos.append(file)

# filename = st.sidebar.text_input('Image name', 'PI_E2_36_5_masked_algatot.bmp')

filename = st.sidebar.selectbox('Select image', archivos)
option_colour = st.sidebar.selectbox('Colour model', ('HSV', 'RGB'))

#path_save = st.sidebar.text_input('Path save name', '2022-06-07/Resultados/')
#analysis_button = st.sidebar.button('Analysis full path')




cola, colb, colc, cold = st.sidebar.columns(4)
with cola: height_1 = st.text_input('Height 1', '800')
with colb: width_1 = st.text_input('Width 1', '1400')
with colc: height_2 = st.text_input('Height 2', '1900')
with cold: width_2 = st.text_input('Width 2', '2500')


# type_analysis = st.sidebar.radio("Analysis type", ('Total teja', 'Microalgas total', 'Microalgas intensas'))





img_raw = cv2.imread(basepath + '\\' + path_name + "\\" + filename)

# alto x ancho
# img = img[400:2500,1000:2900,:]
img = img_raw[int(height_1):int(height_2),int(width_1):int(width_2),:]

plaqueta = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


tab1, tab2 = st.tabs(["Analysis", "Image"])

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
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    fig = px.imshow(img_rgb)
        
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        dragmode = "zoom",
        autosize=False,
        width=1500,
        height=800,)
    
    st.plotly_chart(fig, True)
            

