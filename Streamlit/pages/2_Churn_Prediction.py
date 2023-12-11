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
from PIL import Image
from streamlit_shap import st_shap

# config some settings
pd.set_option('display.max_columns', None) #show all columns

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import self defined functions
from churn_helpers import *

# define subfolder for images
st.session_state.image_folder = 'images/streamlit/'

# create var to print output
print_output = StreamlitPrintOutput()

#set title & configs
st.set_page_config(page_title="Churn Prediction",layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

with st.sidebar:
    add_radio = st.radio(
        "Sections:",
        ("Introduction","Import & DataViz", "Preprocessing", "Modelling", "SHAP", "Probabilities")
    )

if add_radio == "Introduction":
    
    st.header("Introduction to Churn prediction")
    
    # Test to get print feedback back in streamlit
    st.write('This project handles real data from international health insurance contracts.')
    st.write('The goal of this sub-project is to predict if (/probability that) a contract will be terminated by the customer. In particular:')
    st.write('1. Which contrcats have the highest probability to get terminated by the customer?')
    st.write('2. What are main global impacts on contract terminations?')
    st.write('3. What are individual factors for customers to terminate their contracts?')
    
    # Open image of CM
    image_path = st.session_state.image_folder + 'ConfusionMatrix.png'
    try:
        image = Image.open(image_path)
        st.image(image, caption='ConfusionMatrix as description of the classification problem')
    except FileNotFoundError:
        st.error(f"Image file '{image_path}' not found. Please check the file path.")
        
    st.markdown('More information about the project can be found in __[github](https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/tree/main/ChurnProject)__.')
    

if add_radio == "Import & DataViz":
    
    st.header("Import, Data Exploration and Visualization")
    
     
    # Step 1: Import
    st.header("Step 1: Import")
    
    col1, col2 = st.columns([1,1])
    
    # Initialize session state
    if "contracts_loaded" not in st.session_state:
        st.session_state.contracts_loaded = False
    if "products_loaded" not in st.session_state:
        st.session_state.products_loaded = False


    if col1.button("Load Contracts") or st.session_state.contracts_loaded:
         # set 'contracts_loaded' to True
         st.session_state.contracts_loaded = True
    
         st.session_state.contracts = transform_dtypes(extract_contracts())
         col1.write("Contracts DataFrame:")
         # print info
         buffer = io.StringIO()
         st.session_state.contracts.info(buf=buffer)
         s = buffer.getvalue()
         col1.text(s)
         # print head
         col1.dataframe(st.session_state.contracts)
    
    if col2.button("Load Products") or st.session_state.products_loaded:
         # set 'products_loaded' to True
         st.session_state.products_loaded = True
          
         st.session_state.products = get_products()
         col2.write("Products DataFrame:")
         # print info
         buffer = io.StringIO()
         st.session_state.products.info(buf=buffer)
         s = buffer.getvalue()
         col2.text(s)
         # print head
         col2.dataframe(st.session_state.products)

    if st.session_state.contracts_loaded:        
        # Step 2: Dataviz
        st.header("Step 2: DataViz")
        
        st.subheader('Distributions')
        
        st.write('A lot of data explorations & visualizations can be found in the report as well as the 2 notebooks in __[github](https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/tree/main/ChurnProject)__.')
        st.write('Here are only some possibility added. E.g. show column distributions.')
     
        # Dropdown to select contracts column
        col = st.selectbox("Select column to show distribution:", st.session_state.contracts.columns)
        
        # Radio button to select plot type
        plot_type = st.radio("Select plot type:", ["Histogram", "Bar Plot"])
        
        if col and st.button("Show Distribution"):
            # Plotting the distribution based on the selected plot type
            fig, ax = plt.subplots(figsize=(8, 4))
            
            if plot_type == "Histogram":
                sns.histplot(data=st.session_state.contracts, x=col, hue='terminated', multiple="stack", ax=ax)
                ax.set_title(f'Distribution of {col}')
            elif plot_type == "Bar Plot":
                sns.countplot(x=st.session_state.contracts[col], ax=ax)
                ax.set_title(f'Bar Plot of {col}')
            
            ax.set_xlabel(col)
            ax.set_xticklabels(ax.get_xticklabels(),rotation=45, fontsize=10)
            ax.set_ylabel('Frequency')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
        
            # Display the plot in Streamlit
            st.pyplot(fig)
        
            # You can also display other information like descriptive statistics
            st.write(f"Descriptive Statistics for {col}:")
            st.write(st.session_state.contracts[col].describe())
            
        st.subheader('Correlations')
        st.write('Correlations with target variable')
        
        # Dropdown to select the number of features
        num_features = st.selectbox("Select the number of features to use:", range(1, len(st.session_state.contracts.columns) + 1))
        
        # preload df        
        df_initial = transform_dtypes(extract_contracts())
        df_preprocessed = transform_contracts(extract_contracts(), print_terminal=False)
        
        if num_features and st.button('Show Correlations with target variable'):
            
            col1, col2, col3 = st.columns([1,1,1])
            # Plot the correlations for initial, encoded df
            col1.markdown("initial, encoded df:")
            plot_target_corr(df=df_initial, target_col='terminated', k=num_features, encode=True)
            col1.pyplot()
            
            # Plot the correlations for preprocessed, encoded df
            col2.markdown("preprocessed, encoded df:")
            plot_target_corr(df=df_preprocessed.drop(columns=['ds_terminated']), target_col='terminated', k=num_features, encode=True)
            col2.pyplot()
            
            # Plot the correlations with alternative target variable
            col3.markdown("alternative target variable:")
            plot_target_corr(df=df_preprocessed.drop(columns=['terminated']), target_col='ds_terminated', k=num_features, encode=True)
            col3.pyplot()
           

if add_radio == "Preprocessing":
   
   st.header("Preprocessing")
   
   st.subheader('Import')
   
   #preload contracts df
   if "contracts" not in st.session_state:
       st.session_state.contracts = extract_contracts()

   def preprocess_data(extract_contracts, year_only, drop_cols, claim_ratios, add_products, save_csv, cut_effEnd , cut_date ):
       contracts_prepro = transform_df(extract_contracts(), year_only=year_only, drop_cols=drop_cols, claim_ratios=claim_ratios, cut_effEnd = cut_effEnd, cut_date = cut_date)
   
       if add_products:
           products = get_products()
           products = transform_products(products, drop_cols=['product_groupName'])
           contracts_prepro = merge_products_to_contracts(products, contracts_prepro)
   
       if save_csv:
           if file_name:
               save_df(contracts_prepro, filename=file_name)
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
   if "products_loaded" not in st.session_state:
       st.session_state.products_loaded = False
    
   if st.button("Show Contracts"):# or st.session_state.contracts_loaded:
       # set 'contracts_loaded' to True
       st.session_state.contracts_loaded = True
              
       st.write('contracts.head():')
       st.dataframe(st.session_state.contracts.head())
       
   if st.session_state.contracts_loaded:
       #Sidebar: Preprocessing Options
       with st.sidebar:
            st.header("Preprocessing")
            
            st.subheader('Preprocessing Options')
            st.caption('Select Options & run code by button below.')
            
            # keep date or year only of date columns
            year_only = st.selectbox("Keep only year of date cols:", (False, True))
               
            # select cut date   
            cut_effEnd = st.selectbox("Cut effective End-Date:", (True, False))
            
            if cut_effEnd:
                cut_year = st.slider('Maximum for effEndYear:', 2024, 2099, step=1, value = 2030)  
                cut_date = str(cut_year) + '-12-31'
                #st.write(cut_date) #test
            
            # merge product columns to contracts
            add_products = st.selectbox("Merge products:", (False, True))
            
            # create claims ratios instead of absolute values
            claim_ratios = st.selectbox("Create claims ratios:", (True, False))
            
            # Dropdown to select contracts columns to drop
            columns_excluding_activ = [col for col in st.session_state.contracts.columns if col != 'activ']
            drop_cols = st.multiselect("Select columns to drop:", columns_excluding_activ)
            #drop_cols = st.multiselect("Select columns to drop:", st.session_state.contracts.columns)
            
            print_terminal = st.selectbox("Print terminal output:", (True, False))
            
            save_csv = st.selectbox("Save df:", (False, True))
            
            if save_csv:
                file_name = st.text_input("Optional: Enter file name:")
            
            preprocess_button = st.button("Preprocess")
            
            if preprocess_button:
                st.write('continue with modelling')

       st.subheader('Preprocessing')
        # Step 2: Preprocessing 
       st.caption("Choose Options at the sidebar & click button below")
       if preprocess_button:
              
            st.session_state.contracts_preprocessed = True
            with print_output.capture_print_output():
                st.session_state.contracts_prepro = preprocess_data(
                    extract_contracts, year_only, drop_cols, claim_ratios, add_products, save_csv,cut_effEnd = cut_effEnd, cut_date = cut_date)
            
            # print output, if print_terminal set to True
            if print_terminal:
                print_output.display_output()
                
            if st.session_state.contracts_prepro is not None:
               st.write('Preprocessing successful. Preprocessed contract dataframe:')
               st.write("new columns:\n",st.session_state.contracts_prepro.columns)
               st.write("new df contracts_prepro:")
               st.dataframe(st.session_state.contracts_prepro)


  # Modelling
  
if add_radio == "Modelling":
          
    # Initialize session state
    if "train_test" not in st.session_state:
       st.session_state.train_test = False
    if 'model_sel' not in st.session_state:
       st.session_state.model_sel = False
    if "ds_selection" not in st.session_state:
       st.session_state.ds_selection = False  
    recreate_button, train_test_button = None, None  # Initialize outside the loop
    if "encoder" not in st.session_state:
       st.session_state.encoder = None
    if "encoder_name" not in st.session_state:
       st.session_state.encoder_name = None
    if "ds_reasons" not in st.session_state:
       st.session_state.ds_reasons = None
    if "model" not in st.session_state:
       st.session_state.model = None
    if 'selected_options' not in st.session_state:
       st.session_state.selected_options = None
       
    st.header("Modelling") 
    
    #preload preprocessed contracts df
    if "contracts_prepro" not in st.session_state:
       st.info('preprocess contracts first',icon='ℹ️')
    else:
        if st.button("Show preprocessed df"):
           st.write("contracts_prepro.head():\n",st.session_state.contracts_prepro.head())
        
        #Sidebar: Train-/Test-Split Options
        with st.sidebar:
             st.subheader('Train-/Test-Split')
            
             #st.write('st.session_state.train_test:', st.session_state.train_test)
             #st.write('st.session_state.model_sel:', st.session_state.model_sel)
            
             #if not st.session_state.train_test:# or st.session_state.model_sel:
             if not st.session_state.train_test:
             
                st.write('Select Parameters:')
                
                test_size = st.slider('Test size:', 0.1, 0.5, step=0.05, value = 0.2)       
                
                encoder_name = st.selectbox("Encoder:", ('getDummies','CountFrequency'))
                
                scale = st.selectbox("Normalize features:", (False, True))
                
                split_by_date = st.selectbox("Train-test-split time related:", (True, False))
                
                if split_by_date:
                    date_cols = st.session_state.contracts_prepro.select_dtypes(include='datetime').columns
                    split_date_col = st.selectbox('Date column to split train & test:',date_cols)
                    st.session_state.split_by_date = True
                    st.session_state.split_date_col = split_date_col
                else:
                    st.session_state.split_by_date = False
                    split_date_col = None
                    st.session_state.split_date_col = split_date_col #to avoid errors
                    
                ds_target = st.selectbox("Use only selected termination reasons for target variable:", (False, True))
                ds_reasons = ['10014','10015','10016'] #set default to prevent from error
                
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
                    st.session_state.ds_reasons = ds_reasons
                else:
                    st.session_state.ds_selection = False
                

                
                print_shape = False #set default
                print_terminal = st.selectbox("Print terminal output:", (True, False))
     
                train_test_button = st.button('Create Train- & Test-Data')
                if train_test_button:
                    st.session_state.train_test = True
                    
                    # save selections to session state
                    st.session_state.selected_options = {
                                            'test_size': test_size,
                                            'encoder_name': encoder_name,
                                            'scale': scale,
                                            'split_by_date': split_by_date,
                                            'split_date_col': split_date_col,
                                            'ds_target' : ds_target,
                                            'ds_reasons' : ds_reasons
                                        }
                    
             # possibility to rerun with other settings
             else:
                 st.write('Train- & Test-Data already created.')
                 
                 recreate_button = st.button('Doubleclick to Re-Create')
                 if recreate_button:
                     st.session_state.train_test = False
                     #st.session_state.train_test = not st.session_state.train_test
         
            
         
    # Main area on the right - while Train-/Test-Split
    if st.session_state.ds_selection:
        st.write('Reminder - Overview over possible termination reasons:')
        with print_output.capture_print_output():
            show_term_reasons(extract_contracts())
        st.pyplot(plt)
        
    # create train- & test-data after clicking button
    if train_test_button:# or st.session_state.train_test:
        st.session_state.train_test = True
        st.session_state.ds_selection = False # reset to remove picture afterwards
        st.session_state.topK = False # reset topK reduction of SHAP
        
        # save variables in session state
        st.session_state.scale = scale
        st.session_state.ds_target = ds_target       
        
        
        st.subheader('Train- & Test-Data Creation')
                
        with print_output.capture_print_output():
            # create train & test data
            if encoder_name == 'CountFrequency':
                X_train, X_test, y_train, y_test, encoder = create_train_test(st.session_state.contracts_prepro,
                                                                              test_size = test_size,
                                                                              ds_target = ds_target,
                                                                              ds_reasons = ds_reasons,
                                                                              encoder = encoder_name,
                                                                              print_shape = print_shape,
                                                                              split_by_date = split_by_date,
                                                                              split_date_col = split_date_col)
                st.session_state.encoder = encoder
                st.session_state.encoder_name = encoder_name
            else:
                X_train, X_test, y_train, y_test = create_train_test(st.session_state.contracts_prepro,
                                                                              test_size = test_size,
                                                                              ds_target = ds_target,
                                                                              ds_reasons = ds_reasons,
                                                                              encoder = encoder_name,
                                                                              print_shape = print_shape,
                                                                              split_by_date = split_by_date,
                                                                              split_date_col = split_date_col)
                st.session_state.encoder = None
                st.session_state.encoder_name = encoder_name
        
            # Normalize if scale is True
            if scale:
                X_train, X_test = normalize(X_train, X_test)
                
            # seperate and save 'activ'-cols, if existing
            st.session_state.train_activ = X_train['activ']
            st.session_state.test_activ = X_test['activ']
            
            # drop activ col
            X_train.drop(columns=['activ'], inplace = True)
            X_test.drop(columns=['activ'], inplace = True)
                            
        
        # print output, if print_terminal set to True
        if print_terminal:
            print_output.display_output()
    
        # Save results in session_state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        # zip TrainTestData
        st.session_state.TrainTestData = (X_train, X_test, y_train, y_test)
    
        
        st.write("Shapes: X_train:", st.session_state.X_train.shape if X_train is not None else "Data not available", "|",
         "X_test:", st.session_state.X_test.shape if X_test is not None else "Data not available", "|",
         "y_train:", st.session_state.y_train.shape if y_train is not None else "Data not available", "|",
         "y_test:", st.session_state.y_test.shape if y_test is not None else "Data not available")
        
        # show head of train & test data
        col1, col2 = st.columns([3, 1])
        
        n = 5
        # Features
        col1.write("X_train.head():")
        col1.write(st.session_state.X_train[:n])
        col1.write("\nX_test.head():")
        col1.write(st.session_state.X_test[:n])
        
        # Target
        col2.write("y_train.head():")
        col2.write(st.session_state.y_train[:n])
        col2.write("\ny_test.head():")
        col2.write(st.session_state.y_test[:n])
        
    if (train_test_button or st.session_state.train_test):
        st.write("Train and test data created & stored in session_state variables.")
        col1, col2 = st.columns((1,3))
        
        col1.write('selected options:')
        col1.write(st.session_state.selected_options)
        
        train_test_dist(X_train = st.session_state.X_train, X_test = st.session_state.X_test, col = 'start_Year')
        col2.pyplot()
        

    # Modelling Part
    
    if train_test_button or st.session_state.model_sel:
        st.session_state.model_sel = True
        st.session_state.ds_selection = False # reset to remove picture afterwards
                
        #Sidebar: Modelling Options
        with st.sidebar:         
            
            # Tests
            #st.write(st.session_state.train_test)
            #st.write(st.session_state.model_sel)
            
            st.subheader('Model Selection')
         
            selected_model = st.selectbox("Select Classifier:", ["XGBoost", "RandomForest", "DecisionTree"])
            
            select_parameters = st.selectbox("Select Parameters:", ['best', 'individual'])
         
            if selected_model == 'XGBoost':
                sel_model = xgb.XGBClassifier()
                
                if select_parameters == 'individual':
                   # Sample parameters
                    parameter_options = {
                        'learning_rate': [0.005, 0.01, 0.1],
                        'scale_pos_weight': [1, 10, 100],
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 4, 5],
                        'min_child_weight': [1, 3, 5],
                        'reg_alpha': [10**i for i in range(-2, 3)]
                    }
                    
                    # Multiselect boxes for parameters
                    selected_learning_rate = st.selectbox('Select Learning Rate:', parameter_options['learning_rate'])
                    selected_scale_pos_weight = st.selectbox('Select Scale Pos Weight:', parameter_options['scale_pos_weight'])
                    selected_n_estimators = st.selectbox('Select N Estimators:', parameter_options['n_estimators'])
                    selected_max_depth = st.selectbox('Select Max Depth:', parameter_options['max_depth'])
                    selected_min_child_weight = st.selectbox('Select Min Child Weight:', parameter_options['min_child_weight'])
                    selected_reg_alpha = st.selectbox('Select Reg Alpha:', parameter_options['reg_alpha'])
                
                else:
                    # load additional infos
                    grid_params = load_variable('grid_params_XGBClassifier')
                    st.write('best parameters from GridSearch:', grid_params)
                    
         
            elif selected_model == 'RandomForest':
               sel_model = RandomForestClassifier(random_state=42)
    
               if select_parameters == 'individual':
                   # Sample parameters
                   parameter_options = {
                       'n_estimators': [400, 500, 600],
                       'max_depth': [None, 5, 10, 20],
                       'max_features': ['auto', 'sqrt', 'log2']
                   }
    
                   # Multiselect boxes for parameters
                   selected_n_estimators_rf = st.selectbox('Select N Estimators:', parameter_options['n_estimators'])
                   selected_max_depth_rf = st.selectbox('Select Max Depth:', parameter_options['max_depth'])
                   selected_max_features_rf = st.selectbox('Select Max Features:', parameter_options['max_features'])
    
               else:
                   # load additional infos
                   grid_params_rf = load_variable('grid_params_RandomForestClassifier')
                   st.write('best parameters from GridSearch:', grid_params_rf)

            elif selected_model == 'DecisionTree':
               sel_model = DecisionTreeClassifier(random_state=42)
    
               if select_parameters == 'individual':
                    # Sample parameters
                   parameter_options = {'max_depth' : list(range(2, 20))}

                   # Multiselect boxes for parameters
                   selected_max_depth_dt = st.selectbox('Select max_depth:', parameter_options['max_depth'])
                   
               else:
                   params_dt = {'max_depth' : 7}
                   st.write('best parameter from GridSearch:',params_dt)
                   
         
            model_button = st.button("Run & Evaluate Model")
                
            
        # Main Field on the right
        
        # Model definition
        if selected_model == 'XGBoost':
            
            if select_parameters == 'best':
                
                # load best model from GridSearch
                model = load_variable('best_model_XGBClassifier')
                
                # load additional infos
                #f1_grid = load_variable('f1_grid_XGBClassifier')
                #f1_train = load_variable('f1_train_XGBClassifier')
                #f1_test = load_variable('f1_test_XGBClassifier')
                #grid_time = load_variable('grid_time_XGBClassifier')
            
            else:
                # Create XGBClassifier with selected parameters
                model = xgb.XGBClassifier(
                    learning_rate=selected_learning_rate,
                    scale_pos_weight=selected_scale_pos_weight,
                    n_estimators=selected_n_estimators,
                    max_depth=selected_max_depth,
                    min_child_weight=selected_min_child_weight,
                    reg_alpha=selected_reg_alpha
                )
        
        
        elif selected_model == 'RandomForest':
            if select_parameters == 'best':
                # Load best model from GridSearch
                model = load_variable('best_model_RandomForestClassifier')

            else:
                # Create RandomForestClassifier with selected parameters
                model = RandomForestClassifier(
                    n_estimators=selected_n_estimators_rf,
                    max_depth=selected_max_depth_rf,
                    max_features=selected_max_features_rf,
                    random_state=42
                )

        elif selected_model == 'DecisionTree':
            if select_parameters == 'best':
                # Load best model from GridSearch
                model = DecisionTreeClassifier(max_depth=7)

            else:
                # Create RandomForestClassifier with selected parameters
                model = DecisionTreeClassifier(max_depth = selected_max_depth_dt)
                
        
        
        # Evaluations
        if model_button:
            
            st.subheader('Model Evaluation')
            
            #change session state variables
            st.session_state.train_test = False
            st.session_state.model_sel = True
            
            # get variables from session state
            encoder = st.session_state.encoder
            scale = st.session_state.scale 
            ds_target = st.session_state.ds_target 
            ds_reasons = st.session_state.ds_reasons 
            
            # define function to show set parameters
            def display_params(model):
                default_params = sel_model.get_params()
                set_params = model.get_params()
                param_str = "\n".join([f"{param} = {value}," for param, value in set_params.items() if value != default_params[param]])
                return param_str[:-1]
            
            st.write('selected model:', model.__class__.__name__,'(',display_params(model),')')
            
            col1, col2 = st.columns((1,1))
            
            with print_output.capture_print_output():
               #load data from session_state
               X_train = st.session_state.X_train
               X_test = st.session_state.X_test
               y_train = st.session_state.y_train
               y_test = st.session_state.y_test
               
               # fit model
               model.fit(X_train, y_train)
               
               # create predictions
               y_pred_train = model.predict(X_train)
               y_pred_test = model.predict(X_test)
               
               # Print Classification Reports
               col1.text('Classification Report for train data:\n\n')
               col1.text(classification_report(y_train, y_pred_train))
               col1.text('_________________________________________________________\n')
               col1.text('Classification Report for test data:\n\n')
               col1.text(classification_report(y_test, y_pred_test))
               f1_test = f1_score(y_test, y_pred_test).round(2)
               #y_pred_train, y_pred_test, f1_test = eval_model_st(model, data= 'all', X_train = X_train, X_test=X_test, y_train = y_train ,y_test=y_test, norm_CM=None,split_by_date =st.session_state.split_by_date, encoder = encoder, scale = scale, ds_target = ds_target, ds_reasons = ds_reasons, model_infos = False)
            
               print_output.display_output()
               col1.write(f'**F1 Score on test set = {f1_test}**')
               
            #Plot CM
            fig, axes = plt.subplots(2,1, sharex=True)
            plt.suptitle('Confusion Matrices:')
            ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, normalize=None, ax=axes[0])
            axes[0].set_title(f'Train Data')
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize=None, ax = axes[1])
            axes[1].set_title(f'Test Data')
            
            col2.pyplot()
            
            # save model to session_state
            st.session_state.model = model
        
############################################################################
           
 #SHAP (Tests)              
if add_radio == "SHAP":  
    
    # get model to session_state
    model = st.session_state.model
    
    
    ## get data from session_state
    #X_train = st.session_state.X_train
    #X_test = st.session_state.X_test
    #y_train = st.session_state.y_train
    #y_test = st.session_state.y_test
    
    
    #initialize session state
    if 'SHAP' not in st.session_state:
        st.session_state.SHAP = False
    if 'topK' not in st.session_state:
        st.session_state.topK = False
    if 'waterfall' not in st.session_state:
        st.session_state.waterfall = False
    if 'SHAP_sum' not in st.session_state:
        st.session_state.SHAP_sum = False      
    st.session_state.X_train_k, st.session_state.X_test_k = None, None
    if 'explainer' not in st.session_state:
        st.session_state.explainer = shap.Explainer(model)
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = st.session_state.explainer(st.session_state.X_test)    
        
    
    #Parameter Selection at sidebar
    with st.sidebar:
                
        st.subheader('SHAP')
        
        #select parameters for SHAP creation
        st.write('Create Explainer & shap_values:')
        # Create a dictionary to map string options to variables
        data_options = {'X_test': st.session_state.X_test,'X_train': st.session_state.X_train}
        
        # Create a streamlit selectbox
        shap_data_key = st.selectbox('Select data for SHAP explanations:', list(data_options.keys()))
        
        # Access the selected variable based on the key
        shap_data = data_options[shap_data_key]
    
        create_shap = st.button('Create SHAP data')
        
        ######################################        

        # Optional
        st.subheader('Optional: Reduce Features')
        st.write('Recreate train- & test-data with top k features only:')
         
        n = len(st.session_state.X_train.columns)
        k = st.slider('Number of top features to select for SHAP:', 2, n, step=1, value = n)
        run_topK = st.button('select top k features')
        
        ###################################### 
        
        
    # Main window at the right
        
    st.header('SHAP Explanations')
    
    ######################################
    if run_topK:
        st.session_state.topK = True
        
        # Recreate X_train, X_test & fit model again
        X_train, X_test = select_top_k_features(st.session_state.model,X_train = st.session_state.X_train, X_test = st.session_state.X_test, y_train = st.session_state.y_train, y_test = st.session_state.y_test, k = k ,Explainer = 'Explainer')
        
        #refit model
        st.session_state.model.fit(X_train, st.session_state.y_train)

        # update session_state
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        
    if run_topK or st.session_state.topK:
        st.subheader('Reduced Train- & Test-Data')
        
        k_features = st.session_state.X_train.columns
        st.write(f'X_train & X_test got reduced to main {len(k_features)} features:')
        
        col1, col2, col3 = st.columns([1, 3, 1])
 
        
        col1.write(f'top {k} features:')
        col1.write(k_features)
        
        n = 3
        # Features
        col2.write("X_train.head():")
        col2.write(st.session_state.X_train[:n])
        col2.write("\nX_test.head():")
        col2.write(st.session_state.X_test[:n])
        
        # Target
        col3.write("\ny_train.head():")
        col3.write(st.session_state.y_train[:n])
        col3.write("\ny_test.head():")
        col3.write(st.session_state.y_test[:n])
        
        
    #if st.session_state.SHAP and st.session_state.X_train_k:
                  
        

    ######################################
    

    
    #create explainer & SHAP values
    if create_shap:
        st.session_state.SHAP = True
        start_time = time.time()
        
        st.session_state.explainer = shap.Explainer(model)
        st.session_state.shap_values  = st.session_state.explainer(shap_data)      
        
        # reshape shap values, if they are 3dim
        if len(st.session_state.shap_values.shape) == 3:
            shap_values_3d = st.session_state.shap_values
            # Extract the first column of SHAP values
            shap_values_first_col = shap_values_3d.values[:, :, 0]
            base_values_first_col = shap_values_3d.base_values[:, 0]
            st.session_state.shap_values = shap.Explanation(shap_values_first_col, base_values=base_values_first_col, data=shap_data)
            
            
        duration = np.round(time.time() - start_time)
        st.write(f'Explainer & shap_values created in: {duration}s')


    # Show SHAP 
    st.subheader('Feature Importance')    
    col1, col2 = st.columns((1,4))
    if st.session_state.SHAP:
        # Show SHAP                 
        col1.write('Show importance of features based on SHAP values.')
        
        shap_plot  = col1.selectbox('Select kind of SHAP plot:',('summary', 'bar')) #, 'beeswarm'
        n = len(st.session_state.X_train.columns)
        num_features = col1.slider('Number of top features to show:', 2, n, step=1, value = 7)
        
        #optional, don't set first
        shap_example = None 
        Explainer = 'Explainer'
        
        #show/hide summary
        shap_sum = col1.button('show/hide SHAP summary')
        if shap_sum:
            st.session_state.SHAP_sum = not st.session_state.SHAP_sum

 
        # anoter test using repo: https://github.com/snehankekre/streamlit-shap
        if st.session_state.SHAP_sum:
            #st.session_state.SHAP_sum = True
            #reload data from session_state
            shap_values = st.session_state.shap_values
            
            #st.session_state.waterfall = False
            
            # Get the column names
            #column_names = st.session_state.X_test.columns

            # Create SHAP summary plot
            if shap_plot == 'summary':
                splot = shap.summary_plot(shap_values, shap_data, max_display=num_features)  # Adjust X with your feature matrix
            elif shap_plot == 'bar':
                # most important features
                splot = shap.plots.bar(shap_values, max_display=num_features)
                #splot = shap.bar_plot(shap_values, max_display=num_features, show = False)
                
            elif shap_plot == 'beeswarm':
                splot = shap.plots.beeswarm(shap_values, max_display=num_features)
        
            # Display the plot in Streamlit column
            #col1, col2 = st.beta_columns(2)
            col2.pyplot(splot)

        ######################################
        
        # Waterfall plots
        st.subheader('Waterfall plots')
        col1, col2 = st.columns((1,2))
        col1.write('Check indivual reasons for prediction:')
        waterfall_button = col2.button('show/hide SHAP interactive waterfall')
        if waterfall_button:        
            st.session_state.waterfall = not st.session_state.waterfall 

        # Waterfall plots
        if st.session_state.waterfall:
            start = time.time()
            
            col1, col2 = st.columns((2,1))
            M = len(shap_data)
            x = col1.slider('Waterfall example row:', 0, M-1, step=1, value = 0)
            update_waterfall_plot(x, shap_values = st.session_state.shap_values)
            col1.pyplot(bbox_inches='tight', use_container_width = True)
    
            # get computing time
            end = time.time()
            #st.write('execution time:',np.round(end-start),'s')
             
            
        ######################################
        

if add_radio == "Probabilities":
    
    #initialize session state
    if 'proba' not in st.session_state:
        st.session_state.proba = False  
    if 'wf' not in st.session_state:
        st.session_state.wf = False  
    cols = None
        
    # get model to session_state
    model = st.session_state.model
    
    
    # get data from session_state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    
    
    with st.sidebar:
        st.subheader('Probabilities')
        st.write('Select n and get top n contracts of test data based on termination probability')
        
        steps = 10
        n = st.slider('Select n', 10, (st.session_state.X_test.shape[0] // steps) *steps, step = steps, value = 100)
        
        activ = st.selectbox("Select only from activ contracts:", (True, False))

        proba_button = st.button('Get Probabilities')
    
    
    # Main Field 
    st.header('Probabilities')
    
    #st.write(st.session_state.test_activ)
    #st.write(X_test['activ'])
    #st.write(X_test)
    
    if proba_button:
        st.session_state.proba = True
        
        # load activ-infos, if not in columns
        if activ:
            X_train['activ'] = st.session_state.train_activ
            X_test['activ'] = st.session_state.test_activ
        
        # calculate top n contracts with highest pribability
        top_n_terminated, X_test_top_n = predict_top_n_terminated_contracts(model = model, input_df = X_test, n=n, active_only=activ)
        
        # recreate shap_values for top n contracts
        explainer = shap.TreeExplainer(model=model) 
        shap_values_top_n = explainer(X_test_top_n)
        st.session_state.shap_values_top_n = reshape_shap_values(shap_values_top_n)

        #save to session_state
        st.session_state.top_n_terminated, st.session_state.X_test_top_n = top_n_terminated, X_test_top_n 
        
        
    if proba_button or st.session_state.proba:
        st.write(f'{n} contracts with highest termination probability - calculated by {model.__class__.__name__}:')
        
        col1, col2 = st.columns((1,3))
        
        col1.write('termination probabilities:')
        col1.dataframe(st.session_state.top_n_terminated)
        
        col2.write('feature values:')
        col2.dataframe(st.session_state.X_test_top_n)

        if col1.button('Show Waterfall plots') or st.session_state.wf:
            st.session_state.wf = True
            
            st.subheader(f'SHAP Waterfall plot for top {n} contracts')
            
            col3, col4 = st.columns((2,1)) #reduce plot size
            # show interactive waterfall plot
            x = col3.slider('Waterfall example row:', 0, n-1, step=1, value = 0)
            update_waterfall_plot(x, shap_values = st.session_state.shap_values_top_n)
            col3.pyplot(bbox_inches='tight', use_container_width = True)

   
        