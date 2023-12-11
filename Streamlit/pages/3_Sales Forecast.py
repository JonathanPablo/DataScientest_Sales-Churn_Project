# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:07:52 2023

@author: rumiya
"""
import streamlit as st

st.set_page_config(page_title="Sales Forecast",layout="wide")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import helpers_time_series as hts

import os



df = pd.read_csv('SalesData_preprocessed.csv',encoding='latin-1')
df_2 = pd.read_csv('df_pre.csv',low_memory=False)

from helpers import groupby_and_apply, plot_grouped_data, plot_line_plots

st.set_option('deprecation.showPyplotGlobalUse', False)

#Add radio sidebar showing the DataViz and the Modelling

with st.sidebar:
    add_radio = st.radio(
        "Choose:",
        ("Intro","DataViz", "Modelling")
    )

if add_radio == "Intro":
    st.header('Sales Forecast')
    st.write('The goal of this sub-project is to predict the future sales amount within a specific health insurance product.')
    st.write('Click on Subsections in the sidebar for more informations.')

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
   zone_options = ['Old Model Zone', 'New Model Zone']
   model_options = ['SARIMA', 'Multiple Linear Regression', 'Random Forest Regressor', 'XGBoost']
   output_options = ['Metrics and Diagramm', 'Forecast 6 months']
    
   model_options_class = ['KNN', 'SVM']
   metric_options = ['Chebyshev', 'Manhattan', 'Minkowski']
   optimizing_options_class= ['Yes', 'No']
   output_options_class = ['Metrics', 'Forecast']
    
       
   # Level 1: Time series and classification
   project_choice = st.selectbox('Select the project:', projects)
    
   if project_choice == 'Time Series':
       st.subheader('Time Series Options:')
        
       zone_united = st.selectbox('Choose Zone', zone_options)
       model = st.selectbox('Select model:', model_options)
       output = st.selectbox('Output:', output_options)
    
       if st.button('Analyze'):
           if zone_united == 'Old Model Zone':

                df_2 = hts.choose_Zone(df_2, zoneUnited=None)
                df_grouped = hts.group_df_per_month(df_2, '2023-01-01')
                #image = Image.open('../plotly/old_model_zone.png')
                #st.write(df_grouped.head())
                if model != "SARIMA":

                    if model == "Multiple Linear Regression":
                        if output == "Metrics and Diagramm":
                            st.subheader("Results:")
                            st.write("Zone: " + zone_united)
                            st.write("Model: " + model)
                            lag_count = 12
                            df_features = hts.choose_features(df_grouped, 'all', lag_count=lag_count)
                            df_features = df_features.dropna()
                            st.write("Data with Features: ")
                            st.write(df_features.head())
                            st.write("Train Data 2015-2020")
                            st.write("Train Data 2021-2022")
                            train, test = hts.train_test(df_features, '2021-01-01')
                            mlr_model, X_train, y_train, X_test, y_test = hts.multiple_linear_regression(train, test, 'premiumAmount')
                            y_train_pred = mlr_model.predict(X_train)
                            y_test_pred = mlr_model.predict(X_test)
                            evaluation_results = hts.evaluate_regression_model(mlr_model, X_train, y_train, X_test, y_test)
                            st.write(evaluation_results)
                            image = Image.open('plotly/ml_mlr.png')
                            st.image(image, caption='Multiple Lineare Regression')
                        if output == "Forecast 6 months":
                            st.subheader("Forecast:")
                            image = Image.open('plotly/ml_mlr_forecst.png')
                            st.image(image, caption='Random Forest Regressor Forecast')

                    if model == "XGBoost":
                        if output == "Metrics and Diagramm":
                            st.subheader("Results:")
                            st.write("Zone: " + zone_united)
                            st.write("Model: " + model)
                            lag_count = 12
                            df_features = hts.choose_features(df_grouped, 'all', lag_count=lag_count)
                            df_features = df_features.dropna()
                            st.write("Data with Features: ")
                            st.write(df_features.head())
                            st.write("Train Data 2015-2020")
                            st.write("Train Data 2021-2022")
                            train, test = hts.train_test(df_features, '2021-01-01')
                            xgboost_model, X_train, y_train, X_test, y_test = hts.xgboost(train, test, 'premiumAmount')
                            y_train_pred = xgboost_model.predict(X_train)
                            y_test_pred = xgboost_model.predict(X_test)
                            evaluation_results = hts.evaluate_regression_model(xgboost_model, X_train, y_train, X_test, y_test)
                            st.write(evaluation_results)
                            image = Image.open('../plotly/xgboost.png')
                            st.image(image, caption='XGBoost')
                        if output == "Forecast 6 months":
                            st.subheader("Forecast:")
                            image = Image.open('../plotly/xgboost_forecast.png')
                            st.image(image, caption='XGBoost Forecast')

                    if model == "Random Forest Regressor":
                        if output == "Metrics and Diagramm":
                            st.subheader("Results:")
                            st.write("Zone: " + zone_united)
                            st.write("Model: " + model)
                            lag_count = 12
                            df_features = hts.choose_features(df_grouped, 'all', lag_count=lag_count)
                            df_features = df_features.dropna()
                            st.write("Data with Features: ")
                            st.write(df_features.head())
                            st.write("Train Data 2015-2020")
                            st.write("Train Data 2021-2022")
                            train, test = hts.train_test(df_features, '2021-01-01')
                            rf_model, X_train, y_train, X_test, y_test = hts.random_forest_tree(train, test, 'premiumAmount')
                            y_train_pred = rf_model.predict(X_train)
                            y_test_pred = rf_model.predict(X_test)
                            evaluation_results = hts.evaluate_regression_model(rf_model, X_train, y_train, X_test, y_test)
                            st.write(evaluation_results)
                            image = Image.open('../plotly/rf.png')
                            st.image(image, caption='Random Forest Regressor')
                        if output == "Forecast 6 months":
                            st.subheader("Forecast:")
                            image = Image.open('../plotly/rf_forecast.png')
                            st.image(image, caption='Random Forest Regressor Forecast')
                else:
                    import statsmodels.api as sm

                    if output == "Metrics and Diagramm":
                        lag_count = 12
                        df_features = hts.choose_features(df_grouped, 'all', lag_count=lag_count)
                        df_features = df_features.dropna()
                        df_sarima = pd.DataFrame(df_features['premiumAmount'], columns=['premiumAmount'])
                        # st.write(df_sarima.columns)
                        df_sarima_log = hts.stationary_log(df_sarima, kind='log')
                        df_sarima_diff = hts.stationary_log(df_sarima_log, kind='diff')
                        st.subheader("Results:")
                        st.write("Zone: " + zone_united)
                        st.write("Model: " + model)
                        st.write("Log-Transformation = True")
                        st.write("Differencing = 1")
                        st.write("Stationary < 0.05")
                        st.subheader("acf and pacf plot:")
                        train, test = hts.train_test(df_sarima_diff, '2021-01-01')
                        train_sarima = train['premiumAmount']
                        # train_sarima.index = train_sarima.index.to_period('M')
                        test_sarima = test['premiumAmount']
                        st.pyplot(hts.plot_acf_pacf(df_sarima_diff))
                        order = (1, 1, 0)  # (p, d, q)
                        seasonal_order = (1, 1, 0, 12)  # (P, D, Q, S)
                        sarima = sm.tsa.SARIMAX(train_sarima, order=order, seasonal_order=seasonal_order)
                        sarima = sarima.fit(disp=0)
                        y_train_sarima_pred = sarima.predict(train_sarima.index.min(), train_sarima.index.max())
                        y_test_sarima_pred = sarima.predict(test_sarima.index.min(), test_sarima.index.max())
                        evaluation_results = sarima.summary()
                        st.write(evaluation_results)
                        image = Image.open('plotly/sarima.png')
                        st.image(image, caption='SARIMA')
                    if output == "Forecast 6 months":
                        st.subheader("Forecast:")
                        image = Image.open('plotly/sarima_forecast.png')
                        st.image(image, caption='SARIMA Forecast')

           if zone_united == 'New Model Zone':
               df_2 = hts.choose_Zone(df_2, zoneUnited=None)
               df_grouped = hts.group_df_per_month(df_2, '2023-01-01')
               # image = Image.open('old_model_zone.png')
               # st.write(df_grouped.head())
               if model != "SARIMA":

                   if model == "Multiple Linear Regression":
                       if output == "Metrics and Diagramm":
                           st.subheader("Results:")
                           st.write("Zone: " + zone_united)
                           st.write("Model: " + model)
                           lag_count = 12
                           df_features = hts.choose_features(df_grouped, 'all', lag_count=lag_count)
                           df_features = df_features.dropna()
                           st.write("Data with Features: ")
                           st.write(df_features.head())
                           st.write("Train Data 2015-2020")
                           st.write("Train Data 2021-2022")
                           train, test = hts.train_test(df_features, '2021-01-01')
                           mlr_model, X_train, y_train, X_test, y_test = hts.multiple_linear_regression(train, test,
                                                                                                        'premiumAmount')
                           y_train_pred = mlr_model.predict(X_train)
                           y_test_pred = mlr_model.predict(X_test)
                           evaluation_results = hts.evaluate_regression_model(mlr_model, X_train, y_train, X_test,
                                                                              y_test)
                           st.write(evaluation_results)
                           image = Image.open('plotly/ml_mlr_oz.png')
                           st.image(image, caption='Multiple Lineare Regression')
                       if output == "Forecast 6 months":
                           st.subheader("Forecast:")
                           image = Image.open('plotly/ml_mlr_forecst_oz.png')
                           st.image(image, caption='Random Forest Regressor Forecast')

                   if model == "XGBoost":
                       if output == "Metrics and Diagramm":
                           st.subheader("Results:")
                           st.write("Zone: " + zone_united)
                           st.write("Model: " + model)
                           lag_count = 12
                           df_features = hts.choose_features(df_grouped, 'all', lag_count=lag_count)
                           df_features = df_features.dropna()
                           st.write("Data with Features: ")
                           st.write(df_features.head())
                           st.write("Train Data 2015-2020")
                           st.write("Train Data 2021-2022")
                           train, test = hts.train_test(df_features, '2021-01-01')
                           xgboost_model, X_train, y_train, X_test, y_test = hts.xgboost(train, test, 'premiumAmount')
                           y_train_pred = xgboost_model.predict(X_train)
                           y_test_pred = xgboost_model.predict(X_test)
                           evaluation_results = hts.evaluate_regression_model(xgboost_model, X_train, y_train, X_test,
                                                                              y_test)
                           st.write(evaluation_results)
                           image = Image.open('plotly/xgboost_oz.png')
                           st.image(image, caption='XGBoost')
                       if output == "Forecast 6 months":
                           st.subheader("Forecast:")
                           image = Image.open('plotly/xgboost_forecast_oz.png')
                           st.image(image, caption='XGBoost Forecast')

                   if model == "Random Forest Regressor":
                       if output == "Metrics and Diagramm":
                           st.subheader("Results:")
                           st.write("Zone: " + zone_united)
                           st.write("Model: " + model)
                           lag_count = 12
                           df_features = hts.choose_features(df_grouped, 'all', lag_count=lag_count)
                           df_features = df_features.dropna()
                           st.write("Data with Features: ")
                           st.write(df_features.head())
                           st.write("Train Data 2015-2020")
                           st.write("Train Data 2021-2022")
                           train, test = hts.train_test(df_features, '2021-01-01')
                           rf_model, X_train, y_train, X_test, y_test = hts.random_forest_tree(train, test,
                                                                                               'premiumAmount')
                           y_train_pred = rf_model.predict(X_train)
                           y_test_pred = rf_model.predict(X_test)
                           evaluation_results = hts.evaluate_regression_model(rf_model, X_train, y_train, X_test,
                                                                              y_test)
                           st.write(evaluation_results)
                           image = Image.open('plotly/rf_oz.png')
                           st.image(image, caption='Random Forest Regressor')
                       if output == "Forecast 6 months":
                           st.subheader("Forecast:")
                           image = Image.open('plotly/rf_forecast_oz.png')
                           st.image(image, caption='Random Forest Regressor Forecast')
               else:
                   import statsmodels.api as sm

                   if output == "Metrics and Diagramm":
                       lag_count = 12
                       df_features = hts.choose_features(df_grouped, 'all', lag_count=lag_count)
                       df_features = df_features.dropna()
                       df_sarima = pd.DataFrame(df_features['premiumAmount'],columns=['premiumAmount'])
                       #st.write(df_sarima.columns)
                       df_sarima_log = hts.stationary_log(df_sarima, kind='log')
                       df_sarima_diff = hts.stationary_log(df_sarima_log, kind='diff')
                       st.subheader("Results:")
                       st.write("Zone: " + zone_united)
                       st.write("Model: " + model)
                       st.write("Log-Transformation = True")
                       st.write("Differencing = 1")
                       st.write("Stationary < 0.05")
                       st.subheader("acf and pacf plot:")
                       train, test = hts.train_test(df_sarima_diff, '2021-01-01')
                       train_sarima = train['premiumAmount']
                       # train_sarima.index = train_sarima.index.to_period('M')
                       test_sarima = test['premiumAmount']
                       st.pyplot(hts.plot_acf_pacf(df_sarima_diff))
                       order = (1, 1, 0)  # (p, d, q)
                       seasonal_order = (1, 1, 0, 12)  # (P, D, Q, S)
                       sarima = sm.tsa.SARIMAX(train_sarima, order=order, seasonal_order=seasonal_order)
                       sarima = sarima.fit(disp=0)
                       y_train_sarima_pred = sarima.predict(train_sarima.index.min(), train_sarima.index.max())
                       y_test_sarima_pred = sarima.predict(test_sarima.index.min(), test_sarima.index.max())
                       evaluation_results = sarima.summary()
                       st.write(evaluation_results)
                       image = Image.open('plotly/sarima_oz.png')
                       st.image(image, caption='SARIMA')
                   if output == "Forecast 6 months":
                       st.subheader("Forecast:")
                       image = Image.open('plotly/sarima_forecast_oz.png')
                       st.image(image, caption='SARIMA Forecast')




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
   
        
    
    
