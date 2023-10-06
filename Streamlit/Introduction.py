# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 13:31:49 2023

"""
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="SalesData_Project",
    )

st.write("# Sales Data Project")

st.write(
    """
    Course: DS Continious 2023
    
    Participants: Jonathan Leipold, Christian Hirning, Rumiya Al-Meri

    ### About the project:
        - The project deals with the sales data from the broker company 
        which sells health insurance for abroad. 
        - The main target group are the expats who needs the health insurance 
        for a longer time period.
        - The data owner is a BDAE company, the representative is JL.
    
    ### About the product:
        - For this particular project one product (insurance type) was chosen 
        to reduce the complexity.
        - The product price is a premium amount which is paid by a client on a
        monthly, quarterly or a yearly basis.
   
    ### Goals of the project:
        1. Find the best model for forecasting / predicting the premium amount
        2. Find out how premium adjustments impact the value of premium amount  
        3. Find the premium adjustments which maximize the premium amount
        
    ### Data description:
        
        - Two databases are available for analysis: 
            - Sales Data in form of transactions for the period of 
            2014 - 2023YTD, in total about 230 000 transactions
            - Premium adjustments data
        
        - Variables Sales Data:
           
            - Numerical variables:
                AgeAtPremium - Age of Insured Person at PremiumMonth [Years]
                PolicyAgeAtPremium - Age of Contract at PremiumMonth [Years]
                premiumAmount - Target-value - paid premiumAmount (aggregated 
                by Product,CmpPrivate, Deductible, Zone, Time)
                FeeAmount - Fee that the broker (the company) gets from the 
                premium amount (from the insurance company)
               
            - Categorical variables:
                Main-/Sub-ProductCode - Product
                Deductible - Deductible of Contract
                Nation of Premium/Treatment - Nationality of Insured Person
                Zone-(Model) - Zone of country
                Customer-Type (company/private) - Private(I) or Company(C)
            
            - Time variables:
                BirthDate - birth date of an insured person
                premium_startDate / premium_endDate - month the premium is 
                paid, always for a month
                policy_StartDate - contract start date
                policy_EffEndDate - effective end of contract possibly infinite
                (NULL OR 31.12.2099)
"""
)


st.write(
    """### Data Import and first exploration:"""
)
st.write(
    """#### Dataframe Preview:"""
)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   

#import df. set encoding to 'latin-1' necassary to fit to csv-file
df = pd.read_csv('SalesDate-Example.csv',encoding='latin-1')

pd.set_option('display.max_columns', None) #allow display of all columns

#display df
st.dataframe(df)

#Columns info 
st.write(
    """#### Columns information:"""
)

#Columns Info to output in the app
import io
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

st.text(s)

st.write(
    """ ***Observation***: date columns have dtype 'object' --> change to datetime 
    """
)

#Date columns to convert into date type
from helpers import convert_columns_to_datetime
#help(convert_columns_to_datetime) #Description of the function -> uncomment, if needed

columns_to_convert = ['BirthDate','premium_startDate', 'premium_endDate', 'policy_StartDate', 'policy_EffEndDate']

df = convert_columns_to_datetime(df, columns_to_convert)
st.text(df.dtypes)

st.write(
    """#### Description of numerical columns:"""
)

st.write(df.describe())

st.write(
    """ ***Observations***: 
            1. AgeAtPremium and PolicyAgeAtPremium get a negative value of -1 --> 
            explore transactions with the negative Age
            2. Premium amount and FeeAmount get negative value in case of cancelled 
            transactions or wrongly booked transactions
            3. Zone: -1???
            4. Deductible has only value 0 (this is due to the fact, that we only 
            look at one product and that this product offers only deuctible = 0).
            Drop Deductible column.
    """
)

df.drop(columns=['Deductible'], inplace= True)

st.write(
    """#### Import of additional premiumAdjustment df:
        """
        )
        
st.write(
        """ ***The premium_adjustments_example*** data base contains informaion about 
        past adjustments of premiumAmounts of this product.
        The premium adjustments need to be identified not just by their start 
        date but as well by their product option.
        Depending on product this can be defined by variations in model, zone, 
        age-group, deductible.        
        """
   )

pA = pd.read_csv('premium_adjustments_example.csv', sep=';')

#display df
st.dataframe(pA)

#first informations
st.write(pA.info())
st.write(pA.describe())

#Columns Info to output in the app
import io
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

st.text(s)

st.write(
    """ ***Observation***: We can already see, that there is no variation 
    in Min, Max- and MaxSign-Age. Later a closer look on this df, after the
    preprocessing of the main df.
    """
)

df1 = pd.read_csv('SalesData_preprocessed.csv',encoding='latin-1')

