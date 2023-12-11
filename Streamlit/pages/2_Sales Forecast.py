# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:07:52 2023

@author: rumiya
"""
import streamlit as st

st.set_page_config(page_title="02_Sales Forecast",layout="wide")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import helpers_time_series as hts

import os



df = pd.read_csv('data/SalesData_preprocessed.csv',encoding='latin-1')
df_2 = pd.read_csv('data/df_pre.csv',low_memory=False)

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
   zone_options = ['All Zones','Old Model Zone', 'New Model Zone']

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
       if zone_united =='All Zones':
           old = None
           new = None
       if zone_united == 'Old Model Zone':
            old = st.selectbox('Choose:',df_2['Model_Zone'].unique())
            new = None

       elif zone_united == 'New Model Zone':
            new = st.selectbox('Choose:',df_2['Zone_united'].unique())
            old = None
       df_grouped = hts.group_df_per_month(df_2, timerange='2023-01-01', old=old, new=new)
       st.write(hts.plot_timeseries(df_grouped))
       selected_lag = st.selectbox("Choose Lags:", options=list(range(1, 13)))
       df_features = hts.choose_features(df_grouped, 'all', lag_count=selected_lag)
       df_features = df_features.dropna()

       #st.plotly_chart(hts.plot_timeseries(df_features))
       #st.write(df_features.head())
       unique_years = pd.to_datetime(df_features.index).year.unique()

       # Benutzeroberfläche erstellen
       selected_year = st.selectbox("Train/Test Split Year:", options=unique_years[-4:])

       # Erstelle den ausgewählten String im gewünschten Format 'jahr-01-01'
       selected_date_string = f"{selected_year}-01-01"
       train, test = hts.train_test(df_features, selected_date_string)
       #st.write(train.head())
       model = st.selectbox('Select model:', model_options)
       period = st.selectbox('Forecast Months:', options=list(range(1, 13)))
       #output = st.selectbox('Output:', output_options)

       if st.button('Analyze'):
           if model == 'Multiple Linear Regression':

               model, X_train, y_train, X_test, y_test, best_params = hts.multiple_linear_regression_2(train, test,'premiumAmount')
               results = hts.evaluate_regression_model(model, X_train, y_train, X_test, y_test, best_params)
               y_train_pred = model.predict(X_train)
               y_test_pred = model.predict(X_test)
               st.write('Metrics and Best Params:')
               st.write(results)
               st.write('Forecast:')
               future_pred = hts.forecast_regression(df_features, period=period, lag_count=selected_lag, model=model)
               st.write(future_pred['premiumAmount'])
               st.plotly_chart(hts.graph(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, future_pred=future_pred,title='Forecast Multiple Lineare Regression'))

           if model == 'Random Forest Regressor':
               model, X_train, y_train, X_test, y_test, best_params = hts.random_forest_tree(train, test,'premiumAmount')
               results = hts.evaluate_regression_model(model, X_train, y_train, X_test, y_test, best_params)
               y_train_pred = model.predict(X_train)
               y_test_pred = model.predict(X_test)
               st.write('Metrics and Best Params:')
               st.write(results)
               st.write('Forecast:')
               future_pred = hts.forecast_regression(df_features, period=period, lag_count=selected_lag, model=model)
               st.write(future_pred['premiumAmount'])
               st.plotly_chart(
                   hts.graph(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, future_pred=future_pred,
                             title='Forecast Random Forest Regressor'))

           if model == 'Random Forest Regressor':
                   model, X_train, y_train, X_test, y_test, best_params = hts.random_forest_tree(train, test,
                                                                                                 'premiumAmount')
                   results = hts.evaluate_regression_model(model, X_train, y_train, X_test, y_test, best_params)
                   y_train_pred = model.predict(X_train)
                   y_test_pred = model.predict(X_test)
                   st.write('Metrics and Best Params:')
                   st.write(results)
                   st.write('Forecast:')
                   future_pred = hts.forecast_regression(df_features, period=period, lag_count=selected_lag,
                                                         model=model)
                   st.write(future_pred['premiumAmount'])
                   st.plotly_chart(
                       hts.graph(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, future_pred=future_pred,
                                 title='Forecast Random Forest Regressor'))

           if model == 'XGBoost':
                   model, X_train, y_train, X_test, y_test, best_params = hts.xgboost(train, test,'premiumAmount')
                   results = hts.evaluate_regression_model(model, X_train, y_train, X_test, y_test, best_params)
                   y_train_pred = model.predict(X_train)
                   y_test_pred = model.predict(X_test)
                   st.write('Metrics and Best Params:')
                   st.write(results)
                   st.write('Forecast:')
                   future_pred = hts.forecast_regression(df_features, period=period, lag_count=selected_lag,
                                                         model=model)
                   st.write(future_pred['premiumAmount'])
                   st.plotly_chart(
                       hts.graph(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, future_pred=future_pred,
                                 title='Forecast XGboost'))
           if model == 'SARIMA':
               import statsmodels.api as sm
               df_sarima = df_features['premiumAmount']
               df_sarima = pd.DataFrame(df_sarima)
               st.pyplot(hts.plot_acf_pacf(df_sarima))
               #train, test = hts.train_test(df_sarima, selected_date_string)
               train_sarima, test_sarima = hts.train_test(df_sarima, selected_date_string)
               order = (1, 1, 0)  # (p, d, q)
               seasonal_order = (1, 1, 0, 12)  # (P, D, Q, S)
               sarima = sm.tsa.SARIMAX(train_sarima, order=order, seasonal_order=seasonal_order)
               sarima = sarima.fit(disp=0)
               y_train_sarima_pred = sarima.predict(train_sarima.index.min(), train_sarima.index.max())
               y_test_sarima_pred = sarima.predict(test_sarima.index.min(), test_sarima.index.max())
               st.write(sarima.summary())
               sarima_future_pred = sarima.predict(test_sarima.index.max(),
                                                   test_sarima.index.max() + pd.DateOffset(months=period))

               forecast = pd.DataFrame(sarima_future_pred)
               forecast = forecast.rename(columns={'predicted_mean': 'premiumAmount'})
               st.plotly_chart(hts.graph(train_sarima,train_sarima['premiumAmount'],test_sarima,test_sarima['premiumAmount'],y_train_sarima_pred,y_test_sarima_pred,future_pred=forecast,title='Forecast SARIMA'))






           script_content = """
           
   # Generated Python script based on user selections
    
   # Place your script code here based on the selected options
    
   print("Executing generated Python script...")
        """

   elif project_choice == 'Classification':
       st.subheader('Classification model:')


       def main():

           # File upload section
           uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

           # If a file is uploaded, read it into a DataFrame and display
           if uploaded_file is not None:
               # Check if the file is a CSV
               if uploaded_file.type == 'application/vnd.ms-excel':
                   # Use Pandas to read the CSV file
                   target_df_added = pd.read_csv(uploaded_file, sep=';')

                   # Display the loaded DataFrame
                   st.subheader('Loaded DataFrame:')
                   st.dataframe(target_df_added)
               else:
                   st.warning('Please upload a valid CSV file.')


       if __name__ == '__main__':
           main()

       import streamlit as st
       import pandas as pd
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score

       # Load target_df_added dataframe
       target_df_added = pd.read_csv('data/target_df_added.csv', sep=';')

       # Split the data into features (X) and target (y)
       df_x = target_df_added.drop(['nunique_ContractID', 'sum_premiumAmount', 'bin_freq'], axis=1)
       target = target_df_added['bin_freq']

       # Split the data into training and testing sets
       X_train, X_test, y_train, y_test = train_test_split(df_x, target, test_size=0.2, random_state=42)


       # Streamlit demo
       def main():
           st.subheader('K-Neighbors Classifier Predictions')

           # Sidebar for user input
           st.sidebar.header('Features Input')

           # Input fields for features
           premium_year = int(st.sidebar.slider("Select Premium Year", min_value=target_df_added['year'].min(),
                                                max_value=target_df_added['year'].max(),
                                                value=target_df_added['year'].min()))
           premium_month = int(st.sidebar.slider("Select Premium Month", min_value=1, max_value=12, value=1))
           mean_AgeatPremium = int(
               st.sidebar.slider("Select mean of AgeatPremium", min_value=target_df_added['mean_AgeAtPremium'].min(),
                                 max_value=target_df_added['mean_AgeAtPremium'].max(),
                                 value=target_df_added['mean_AgeAtPremium'].mean()))
           mean_policyAge_months = int(st.sidebar.slider("Select mean of policyAge_months",
                                                         min_value=target_df_added['mean_policyAge_months'].min(),
                                                         max_value=target_df_added['mean_policyAge_months'].max(),
                                                         value=target_df_added['mean_policyAge_months'].mean()))
           sum_premiumAmount_prev12months = int(st.sidebar.slider("Select mean of sum_premiumAmount_prev12months",
                                                                  min_value=target_df_added[
                                                                      'sum_premiumAmount_prev12months'].min(),
                                                                  max_value=target_df_added[
                                                                      'sum_premiumAmount_prev12months'].max(),
                                                                  value=target_df_added[
                                                                      'sum_premiumAmount_prev12months'].mean()))

           # Options for K-neighbors and metric
           k_neighbors_options = [str(i) for i in range(1, 11)]
           k_neighbors = st.selectbox("Select K-neighbors (1-10)", k_neighbors_options)
           metric_options = ['chebyshev', 'manhattan', 'minkowski']
           metric = st.selectbox("Select Metric", metric_options)

           # Button to trigger prediction
           if st.button('Predict bin_freq'):

               try:
                   # Converting selected values to appropriate types
                   premium_year = int(premium_year)
                   premium_month = int(premium_month)
                   mean_AgeatPremium = int(mean_AgeatPremium)
                   mean_policyAge_months = int(mean_policyAge_months)
                   sum_premiumAmount_prev12months = int(sum_premiumAmount_prev12months)
                   k_neighbors = int(k_neighbors)

                   # Preparing input data for prediction
                   input_data = np.array([premium_year, premium_month, mean_AgeatPremium, mean_policyAge_months,
                                          sum_premiumAmount_prev12months]).reshape(1, -1)

                   # Loading K-Neighbors model with user-selected options
                   knn_model = KNeighborsClassifier(n_neighbors=k_neighbors, metric=metric)
                   knn_model.fit(X_train, y_train)

                   # Performing prediction using K-Neighbors model
                   prediction = knn_model.predict(input_data)[0]

                   # Display prediction
                   st.success(f'Predicted bin_freq: {prediction}')

                   # Calculating and displaing accuracy score
                   y_pred = knn_model.predict(X_test)
                   accuracy = accuracy_score(y_test, y_pred)
                   st.info(f'Accuracy Score of the model for test data: {accuracy}')

               except ValueError as e:
                   st.error(f"An error occurred during conversion: {e}")
               except Exception as e:
                   st.error(f"An unexpected error occurred: {e}")


       if __name__ == '__main__':
           main()

   
        
    
    
