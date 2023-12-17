# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 13:31:49 2023

"""
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# define subfolder for images
if 'image_folder' not in st.session_state:
  st.session_state.image_folder = 'images/streamlit/'

st.set_page_config(
    page_title="Sales Forecast & Churn Prediction",layout="wide"
    )

#initialize session_state to keep showing columns later
if 'sales' not in st.session_state:
  st.session_state.sales = False
if 'churn' not in st.session_state:
  st.session_state.churn = False 

st.header("Sales- & Churn-prediction of health-insurance data")

"""
Course: DS Continious 2023

Participants: Jonathan Leipold, Christian Hirning, Rumiya Al-Meri


### Introduction
This demo shows some results of a project during the course 'Data Science Continuous Mar23' of DataScientest. 

The project works with real data from a company specialized on international health insurance products. 

The data comes directly from the companies ERP System and contains contract- as well as premium- & claims-related informations in pseudonymised form.
"""

if st.button('Show data collection process'):
    # Open image of DCprocess
    image_path = st.session_state.image_folder + 'DataCollection.png'
    try:
        image = Image.open(image_path)
        st.image(image, caption='Process of data collection from ERP System to JupyterNotebook')
    except FileNotFoundError:
        st.error(f"Image file '{image_path}' not found. Please check the file path.")

"""
The main objective was to create the best performing model for sales predictions, in particularly prediction of premium amounts per month. Due to the big variety of product characteristics, only transactions concerning one main product type were considered to build a prototype.

During the project, the project group faced the problem of a limited number of features which are known for the future. Therefore, it was decided on project extension with the further objective, namely churn predictions. The contracts’ data for all products was taken and enriched by additional, information from the ERP-System. 
"""

col1, col2 = st.columns((1,1))

####################################
# Sales Informations in left column

if col1.button("Show 1. Sub-Project *Sales Forecast*") or st.session_state.sales:
  st.session_state.sales = True # set to keep showing
  text_sales =  """
                ##### 2 initial main goals of 1. sub-project "*Sales-Forecast*":
                  1.	Find the best model for forecasting / predicting the premium amount
                  2.	Find out how premium adjustments impact the value of premium amount 
                """
  col1.markdown(text_sales)

  # Open image of sales forecast
  image_path = st.session_state.image_folder + 'SalesForecast.png'
  try:
      image = Image.open(image_path)
      col1.image(image, caption='Example of Sales Forecast')
  except FileNotFoundError:
      col1.error(f"Image file '{image_path}' not found. Please check the file path.")

####################################
# Churn informations in second column
if col2.button("Show 2. Sub-Project *Churn Prediction*") or st.session_state.churn:
  st.session_state.churn = True # set to keep showing
  text_churn =  """
                ##### 2 additional main goals of 2. sub-project "*Churn Prediction*":
                  1.	Identify main features that have an impact on customers’ termination behaviour
                  2.	Find active contracts that are more likely to get terminated by the customer
                """

  col2.markdown(text_churn)

  # Open image of churn 
  image_path = st.session_state.image_folder + 'SHAP.png'
  try:
      image = Image.open(image_path)
      col2.image(image, caption='Example of Churn Interpretation')
  except FileNotFoundError:
      col2.error(f"Image file '{image_path}' not found. Please check the file path.")

####################################

"""
More Information can be found in detailled form in the __[Final Report](<https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/blob/main/Sales%20Forecast%20and%20Churn%20Prediction_Final%20Report.docx>)__ and in general on __[github](https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/tree/main)__.

To go to results of the 2 sub-projects click the sections on the left.
"""