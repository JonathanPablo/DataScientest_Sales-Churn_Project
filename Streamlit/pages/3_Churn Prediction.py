# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:08:20 2023

@author: rumiy
"""

import streamlit as st

st.set_page_config(page_title="03_Churn Prediction")

with st.sidebar:
    add_radio = st.radio(
        "Choose:",
        ("DataViz", "Modelling")
    )
    
if add_radio == "DataViz":
    
    st.markdown("# Data Exploration and Visualization")



if add_radio == "Modelling":
   
   st.markdown("# Modelling")

   import pandas as pd
#from preprocessing-Modelling_01 import extract_contracts, get_products

# Step 1: Import
   st.header("Step 1: Import")

   if st.button("Get Contracts"):
       contracts = extract_contracts()
       st.write("Contracts DataFrame:")
       st.write(contracts.head())

   if st.button("Get Products"):
       products = get_products()
       st.write("Products DataFrame:")
       st.write(products.head())

    # Step 2: Preprocessing
   st.header("Step 2: Preprocessing")
    
   keep_year = st.selectbox("Keep only year of date cols:", ("Yes", "No"))
    
   # Dropdown to select contracts columns to drop
   #contracts_cols_to_drop = st.multiselect("Select contracts columns to drop:", contracts.columns)    
   merge_products = st.selectbox("Merge products:", (True, False))
    
    #if merge_products:
    # Dropdown to select products columns to drop
    #products_cols_to_drop = st.multiselect("Select products columns to drop:", products.columns)

   encode_df = st.selectbox("Encode df:", (True, False))

   save_df = st.selectbox("Save df:", (True, False))

   if save_df:
       file_name = st.text_input("Enter file name:")

   if st.button("Preprocess"):
       print('Preprocess')
    # Preprocess data based on selected options
    # Call your preprocessing function with the selected options
    # preprocess_data(contracts, products, keep_year, contracts_cols_to_drop, merge_products, products_cols_to_drop, encode_df)

   # Step 3: Modelling
   st.header("Step 3: Modelling")

   train_test_split_time = st.selectbox("Train-test-split time related:", (True, False))

   selected_model = st.selectbox("Select Model:", ["Decision Tree", "Xgboost"])

   k_folds = st.selectbox("Select parameters:", [2, 3, 4])

   if st.button("Run Model"):
       print('Run the selected model with the chosen parameters')
    