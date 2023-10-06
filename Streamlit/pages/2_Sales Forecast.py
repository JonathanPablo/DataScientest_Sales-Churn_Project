# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:07:52 2023

@author: rumiya
"""
import streamlit as st

st.set_page_config(page_title="02_Sales Forecast")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  

df = pd.read_csv('SalesData_preprocessed.csv',encoding='latin-1')


from helpers import groupby_and_apply, plot_grouped_data, plot_line_plots

st.set_option('deprecation.showPyplotGlobalUse', False)

#Add radio sidebar showing the DataViz and the Modelling

with st.sidebar:
    add_radio = st.radio(
        "Choose:",
        ("DataViz", "Modelling")
    )
    
if add_radio == "DataViz":
    
    st.markdown("# Data Exploration and Visualization")
    
    st.write(
        """#### Target Values:
           """
        )

        
    st.write("""***Create target values:*** sum PremiumAmount, count ContractID
             """
             )

    st.write(""" ***Main preprocessing goal:*** 
             Create df where rows are crouped by months & product options.
             Product options can be:
         1. Different Deductible --> Here only 0 --> column dropped
         2. Different Age groups --> As seen in pA there is no variation
         3. Different Model & Zone --> There is variation, so we need to group by this columns as well.
             """
             )
    
    #Build a timeline with a sum of premiumAmount and ContractID count
    st.write(
        """##### Premium Amount (sum) and ContractID (count) grouped by PremiumMonth:
            """
    )
    
    group_cols = ['premiumMonth']
    apply = {'premiumAmount': 'sum', 'ContractID': 'count'}
    st.pyplot(plot_grouped_data(df, group_cols= group_cols, apply=apply, x='premiumMonth'))

    st.write(
        """ ***Observation***: 1. premiumAmount_sum & ContractID_count keep 
        growing with 1 drop between 2018-2020.
        2. 2020 makes sense due to covid --> less traveling, less demand for 
        international health insurance. 3. Drop of premiumAmount_sum in 
        2018-2019 - additional features will be explored to find the reason for
        this drop
        """
        )
    
    #Build a plot of different regions to explain the drop due to the product range change
        
    group_cols2 = group_cols + ["ZoneDesc"]
    x = 'premiumMonth'
    apply2 = {'ContractID': 'count'}

    df_grouped_pM_and_Zone = groupby_and_apply(df, group_cols= group_cols2, apply= apply2)
    df_grouped_pM_and_Zone.head()

    from helpers import auto_convert_columns_to_datetime
    df_grouped_pM_and_Zone = auto_convert_columns_to_datetime(df_grouped_pM_and_Zone)
    df_grouped_pM_and_Zone.info()
        
    
    st.write(
        """##### The drop between 2018 and 2019 can be explained by the zones change: 
            """
    )
    
    st.pyplot(plot_line_plots(df_grouped_pM_and_Zone, hue= 'ZoneDesc', y=['count_ContractID']))

    st.write(
        """***Observation:*** There was a change in the product structure which made many people cancel their contract.
        ***Conclusion:*** A new ZoneModel seems to have an impact on the Amount of Contracts.
            """
            )

    #PolicyAgeAtPremium - check the distribution of PolicyAge and ContractID count
    from helpers import plot_subplots
    
    st.write(
        """##### PolicyAgeAtPremium vs ContractID counts and PremiumAmount: 
            """
    )
    
    
    group_cols = ['PolicyAgeAtPremium']
    apply = {'ContractID':'count','premiumAmount':'mean',}
    groupby_and_apply(df, group_cols=group_cols, apply=apply)  

    columns = ['premiumAmount','ContractID']
    apply_columns = {'premiumAmount':'mean', 'ContractID' : 'count'}
    st.pyplot(plot_subplots(df, 'PolicyAgeAtPremium', other_columns=columns , functions=apply_columns))     
    
#else:
    


#Exploration of all the columns to decide which and how to use them.
#Use of self-defined functions from the helpers.py file

#Create a Modelling sub-page

if add_radio == "Modelling":
   
   st.markdown("# Modelling")

   # Define options
   projects = ['Time Series', 'Classification']
   yes_no_options = ['Yes', 'No']
   model_options = ['Sarima', 'LR', 'RF', 'XGBoost']
   feature_options = ['Time features', 'Lag', 'Contracts']
   optimizing_options = ['CV', 'GS']
   output_options = ['Metrics', 'Forecast', 'Diagramm']
    
   model_options_class = ['KNN', 'SVM']
   metric_options = ['Chebyshev', 'Manhattan', 'Minkowski']
   optimizing_options_class= ['Yes', 'No']
   output_options_class = ['Metrics', 'Forecast']
    
       
   # Level 1: Time series and classification
   project_choice = st.selectbox('Select the project:', projects)
    
   if project_choice == 'Time Series':
       st.subheader('Model options:')
        
       zone_united = st.selectbox('Zone united:', yes_no_options)
       model = st.selectbox('Select model:', model_options)
       features = st.selectbox('Select features:', feature_options)
       optimizing = st.selectbox('Optimizing:', optimizing_options)
       output = st.selectbox('Output:', output_options)
    
       if st.button('Submit'):
           # Create a Python script based on selected options
           script_content = """
   # Generated Python script based on user selections
    
   # Place your script code here based on the selected options
    
   print("Executing generated Python script...")
        """
    
   elif project_choice == 'Classification':
        st.subheader('Classification model:') 
        
        model_class = st.selectbox('Select model:', model_options_class)
        metrics = st.selectbox('Select metric:', metric_options)
        optimizing_class = st.selectbox('Optimizing:', optimizing_options_class)
        output_class = st.selectbox('Output:', output_options_class)
   
        
    
    
