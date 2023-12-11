# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 13:31:49 2023

"""
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Sales Forecast & Churn Prediction",layout="wide"
    )

st.header("Sales- & Churn-prediction of health-insurance data")

"""
Course: DS Continious 2023

Participants: Jonathan Leipold, Christian Hirning, Rumiya Al-Meri


### Introduction
This demo shows some results of a project during the course 'Data Science Continuous Mar23' of DataScientest. 

The project works with real data from a company specialized on international health insurance products. The data comes directly from the companies ERP System and contains contract- as well as premium- & claims-related informations in pseudonymised form.

The main objective was to create the best performing model for sales predictions, in particularly prediction of premium amounts per month. Due to the big variety of product characteristics, only transactions concerning one main product type were considered to build a prototype.

#### Initially 2 main goals were defined in the first sub-project:
  1.	Find the best model for forecasting / predicting the premium amount
  2.	Find out how premium adjustments impact the value of premium amount 

During the project, the project group faced the problem of a limited number of features which are known for the future. Therefore, it was decided on project extension with the further objective, namely churn predictions. The contracts’ data for all products was taken and enriched by additional, information from the ERP-System. 

#### Withing this second sub-project another 2 goals were defined:
  1.	Identify main features that have an impact on customers’ termination behaviour
  2.	Find active contracts that are more likely to get terminated by the customer

More Information can be found in detailled form in the __[Final Report](<https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/blob/main/Sales%20Forecast%20and%20Churn%20Prediction_Final%20Report.docx>)__.

To go to results of the 2 sub-projects click the sections on the right.
"""