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
        target_df_added = pd.read_csv('target_df_added.csv', sep=';')

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
            premium_year = int(st.sidebar.slider("Select Premium Year", min_value=target_df_added['year'].min(), max_value=target_df_added['year'].max(), value=target_df_added['year'].min()))
            premium_month = int(st.sidebar.slider("Select Premium Month", min_value=1, max_value=12, value=1))
            mean_AgeatPremium = int(st.sidebar.slider("Select mean of AgeatPremium", min_value=target_df_added['mean_AgeAtPremium'].min(), max_value=target_df_added['mean_AgeAtPremium'].max(), value=target_df_added['mean_AgeAtPremium'].mean()))
            mean_policyAge_months = int(st.sidebar.slider("Select mean of policyAge_months", min_value=target_df_added['mean_policyAge_months'].min(), max_value=target_df_added['mean_policyAge_months'].max(), value=target_df_added['mean_policyAge_months'].mean()))
            sum_premiumAmount_prev12months = int(st.sidebar.slider("Select mean of sum_premiumAmount_prev12months", min_value=target_df_added['sum_premiumAmount_prev12months'].min(), max_value=target_df_added['sum_premiumAmount_prev12months'].max(), value=target_df_added['sum_premiumAmount_prev12months'].mean()))
        
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
                    input_data = np.array([premium_year, premium_month, mean_AgeatPremium, mean_policyAge_months, sum_premiumAmount_prev12months]).reshape(1, -1)
            
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
        
        
        
                
