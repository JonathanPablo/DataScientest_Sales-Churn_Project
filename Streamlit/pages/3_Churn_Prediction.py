# -*- coding: utf-8 -*-
"""
Created on 2023-10-15

@author: JL
"""
# if nessesarry install packages
#pip install feature_engine #example 

# import libraries & modules
from IPython import display
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno #NaN-plotting
import requests #to get e.g. country information from REST API
import time

# functions & modules for ML
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, roc_curve, auc
import shap
import xgboost as xgb
from feature_engine.encoding import CountFrequencyEncoder

# import streamlit & Co 
import streamlit as st
import io
import contextlib
import sys

# config some settings
pd.set_option('display.max_columns', None) #show all columns

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import self defined functions
from churn_helpers import *

# create var to print output
print_output = StreamlitPrintOutput()


st.set_page_config(page_title="03_Churn Prediction")

with st.sidebar:
    add_radio = st.radio(
        "Sections:",
        ("Introduction","Import & DataViz", "Preprocessing", "Modelling")
    )

if add_radio == "Introduction":
    
    st.header("Introduction to Churn prediction")
    
    # Test to get print feedback back in streamlit
    def test():
        a=2
        print('test1')
        if a==2:
            print('test2')
        return a
    
    if st.button("Test"):
        with print_output.capture_print_output():
            a = test()
        print_output.display_output()
        st.write(a)
        
    
    

if add_radio == "Import & DataViz":
    
   st.header("Import, Data Exploration and Visualization")

    
   # Step 1: Import
   st.header("Step 1: Import")

   if st.button("Load Contracts"):
       st.session_state.contracts = extract_contracts()
       st.write("Contracts DataFrame:")
       # print info
       buffer = io.StringIO()
       st.session_state.contracts.info(buf=buffer)
       s = buffer.getvalue()
       st.text(s)
       #print head
       st.write(st.session_state.contracts)

   if st.button("Load Products"):
       st.session_state.products = get_products()
       st.write("Products DataFrame:")
       # print info
       buffer = io.StringIO()
       st.session_state.products.info(buf=buffer)
       s = buffer.getvalue()
       st.text(s)
       #print head
       st.write(st.session_state.products)
 

if add_radio == "Preprocessing":
   
   st.header("Preprocessing")
   
   st.subheader('Import')
   
   #preload contracts df
   if "contracts" not in st.session_state:
       st.session_state.contracts = extract_contracts()

   def preprocess_data(extract_contracts, year_only, drop_cols, claim_ratios, add_products, save_csv):
       contracts_prepro = transform_df(extract_contracts(), year_only=year_only, drop_cols=drop_cols, claim_ratios=claim_ratios)
   
       if add_products:
           products = get_products()
           products = transform_products(products, drop_cols=['product_groupName'])
           contracts_prepro = merge_products_to_contracts(products, contracts_prepro)
   
       if save_csv:
           if file_name:
               save_df(contracts_prepro, filename=file_name+str('.csv'))
           else:
               save_df(contracts_prepro)
   
       return contracts_prepro
      
  
   #Main Field on the right
   
   # 1 Import
   
   st.caption("Click Button to load initial contracts dataframe first")
         
   # Initialize session state
   if "contracts_loaded" not in st.session_state:
       st.session_state.contracts_loaded = False
   if 'contracts_preprocessed' not in st.session_state:
       st.session_state.contracts_preprocessed = False
    
   if st.button("Show Contracts") or st.session_state.contracts_loaded:
       # set 'contracts_loaded' to True
       st.session_state.contracts_loaded = True
              
       st.write('contracts:')
       st.dataframe(st.session_state.contracts)
       
       #Sidebar: Preprocessing Options
       with st.sidebar:
           st.header("Preprocessing")
           
           st.subheader('Preprocessing Options')
           st.caption('Select Options & run code by button below.')
           
           # keep date or year only of date columns
           year_only = st.selectbox("Keep only year of date cols:", (False, True))
            
           # merge product columns to contracts
           add_products = st.selectbox("Merge products:", (False, True))
           
           # create claims ratios instead of absolute values
           claim_ratios = st.selectbox("Create claims ratios:", (True, False))
            
           # Dropdown to select contracts columns to drop
           drop_cols = st.multiselect("Select columns to drop:", st.session_state.contracts.columns)
        
           print_terminal = st.selectbox("Print terminal output:", (True, False))
        
           save_csv = st.selectbox("Save df:", (False, True))

           if save_csv:
               file_name = st.text_input("Optional: Enter file name:")

       st.subheader('Preprocessing')
        # Step 2: Preprocessing 
       st.caption("Choose Options at the sidebar & click button below")
       if st.button("Preprocess"):
            st.session_state.contracts_preprocessed = True
            with print_output.capture_print_output():
                st.session_state.contracts_prepro = preprocess_data(
                    extract_contracts, year_only, drop_cols, claim_ratios, add_products, save_csv)
            
            # print output, if print_terminal set to True
            if print_terminal:
                print_output.display_output()
                
            if st.session_state.contracts_prepro is not None:
               st.write('Preprocessing successful. Preprocessed contract dataframe:')
               st.write("new columns:\n",st.session_state.contracts_prepro.columns)
               st.write("new df contracts_prepro:\n",st.session_state.contracts_prepro.head())
            

if add_radio == "Modelling":
          
    # Initialize session state
    if "train_test" not in st.session_state:
       st.session_state.train_test = False
    if 'model_sel' not in st.session_state:
       st.session_state.model_sel = False
    if "ds_selection" not in st.session_state:
       st.session_state.ds_selection = False
       
    st.header("Modelling") 
    
    #preload preprocessed contracts df
    if "contracts_prepro" not in st.session_state:
       st.info('preprocess contracts first',icon = 'i')
    else:
       if st.button("Show preprocessed df"):
           st.write("contracts_prepro.head():\n",st.session_state.contracts_prepro.head())
  
    #Sidebar: Modelling Options
    with st.sidebar:
       
       st.subheader('Train-/Test-Split')
       
       test_size = st.slider('Test size:', 0.1, 0.5, step=0.05)       
       
       split_by_date = st.selectbox("Train-test-split time related:", (True, False))
       
       if split_by_date:
           date_cols = st.session_state.contracts_prepro.select_dtypes(include='datetime').columns
           split_date_col = st.selectbox('Date column to split train & test:',date_cols)
           
       ds_target = st.selectbox("Use only selected termination reasons for target variable:", (False, True))
       
       if ds_target:
           st.session_state.ds_selection = True
           
           #get termination reations sorted
           arr = extract_contracts().terminationReason.unique()
           # Create a new list with only valid integer values
           valid_integers = [x for x in arr if str(x).isnumeric()]
           # Convert the resulting list to integers
           valid_integers = [int(x) for x in valid_integers]
           # Convert the list back to a NumPy array if needed
           term_vals = np.sort(np.array(valid_integers)).astype('str')
           
           ds_reasons = st.multiselect('Selected termination reasons: (recommended: [10014,10015,10016])', term_vals)
           # default: ['10014','10015','10016'] 
           
    
       if st.session_state.train_test:
           
           st.subheader('Model Selection')
        
           selected_model = st.selectbox("Select Model:", ["Decision Tree", "Xgboost"])
        
           k_folds = st.selectbox("Select parameters:", [2, 3, 4])
        
           if st.button("Run Model"):
               print('Run the selected model with the chosen parameters')
   
    # Main area on the right
    if ds_target:
        st.write('Reminder - Overview over possible termination reasons:')
        with print_output.capture_print_output():
            show_term_reasons(extract_contracts())
        st.pyplot(plt)
            