# Use function in notebook in same folder by: from helpers import function
#e.g.: from churn_helpers import country_to_region_mapping 

#_________________________________________________________________________________________________________________________
# import libraries & modules
from IPython import display
import ipywidgets as widgets
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas_profiling  DeprecationWarning: `import pandas_profiling` is going to be deprecated by April 1st. Please use `import ydata_profiling` instead.
from ydata_profiling import ProfileReport #import ydata_profiling
import missingno as msno #NaN-plotting
import requests #to get e.g. country information from REST API
import time
import ast # to print here created functions
import math

# functions & modules for ML
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, fbeta_score, make_scorer
import shap
import xgboost as xgb
from feature_engine.encoding import CountFrequencyEncoder
import scikitplot as skplt


#for resampling imbalanced data:
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

#streamlit & Co
import streamlit as st
import io
import contextlib
import os, sys
import joblib

#_________________________________________________________________________________________________________________________
#initiate variables to avoid erros:
X_train, X_test, y_train, y_test = (0,0,0,0)
prepro_params, split_params, scaling_params, ds_target, filename, reduce_ratio, scale, scaler, eval_data, split_by_date, Explainer, shap_values, cv = None, None, None, None, None, None, None, None, None, None, None, None, None

#_________________________________________________________________________________________________________________________

#create class to hide print outputs
class HiddenPrints:
    '''
    Hides print outputs if used like this:
    
    with HiddenPrints():
        print("This will not be printed")

    print("This will be printed as before")
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#_________________________________________________________________________________________________________________________

# Extract, save, preprocess, dataviz, ...
#_________________________________________________________________________________________________________________________


#Take updates version _v2 from 23.09.2023:
def extract_contracts(version = 'v2'):
    '''
    creates pandas df out of csv-file (default: 'BDAE_DataMining_Policies_v2.csv', must be in same folder) 
    '''
    if version == 'v2':
        contracts = pd.read_csv('BDAE_DataMining_Policies_v2.csv', encoding = 'ISO-8859-1')
    elif version == 'v1':
        contracts = pd.read_csv('BDAE_DataMining_Policies.csv', encoding = 'ISO-8859-1')
    else:
        print('version not available')
    return contracts

#_________________________________________________________________________________________________________________________

# plot NaN Overview
def plot_NaNs(contracts):
    '''
    print & plot overview over ratio of NaN values per column. Graphically and in numbers.
    '''
    
    # plot barplot of NaN-values
    plt.figure(figsize=(15,10))
    sns.barplot(y=contracts.columns, x=contracts.isna().sum())
    plt.xlabel('NaN-Count', fontsize=12)
    plt.title(f'Overview of NaN-Values per column. Total rows: {len(contracts)}', fontsize=16);
    
    #show 10 cols with most % of NaN values
    rows = len(contracts)
    NaNs = ((contracts.isnull().sum(axis=0).sort_values(ascending=False)/rows*100).round(1))
    NaNs = NaNs.loc[NaNs > 0]
    print(f'Number of rows: {rows}\nPercent of NaN-values per column:')
    print(NaNs.head(10))
    
#_________________________________________________________________________________________________________________________

# extra function to transform date columns
def transform_dtypes(df, dateStr = ['Date','date','until']): 
    '''
    loop over all columns & convert all "Date" columns to datatype date
    
    input: 
    - df
    - dateStr : list of strings, if col name contains one of them datatype should be transformed to date (default: ['Date','date','until'])
    
    return: 
    df with date columns transform to datatype date 

    '''
    for column in df.columns:
        if any([x in column for x in dateStr]):
            df[column] = pd.to_datetime(df[column], format='%Y-%m-%d', errors = 'coerce') #set out of range dates to NaT
            if 'EndDate' in column:
                df[column].fillna('2099-12-21', inplace=True) #fill NaT by end of century for EndDates
                
    #convert remaining object cols to str/..
    df = df.convert_dtypes()
    
    # idea: Set index 'ContractID': avoid encoding ContractID, which is the unique key anyway --> set as index col in new df (if not already set)
    #if 'ContractID' in df.columns:
    #    df = df.set_index('ContractID') 
       
    return df

#_________________________________________________________________________________________________________________________

def built_claim_ratios(df):
    '''
    calculate ratio of claimed amount instead of absolute values.
    '''  
    
    # replace NaN in payout cols by 0
    df[['sum_payout_lastActivYear','sum_payout_total']].fillna(0, inplace=True)
    #Alternative, if fillna doesn't work:
    #df['sum_payout_lastActivYear'] = np.where(pd.isna(df.sum_payout_lastActivYear) == True ,0,df.sum_payout_lastActivYear)
    #df['sum_payout_total'] = np.where(pd.isna(df.sum_payout_total) == True ,0,df.sum_payout_total)
    
    # calculate payout ratio (again correctly --> set to 1, if no money was claimed, because "everything" got paid out)
    df['payout_ratio_lastActivYear'] = np.where(df.sum_claimed_lastActivYear == 0 ,1,df.sum_payout_lastActivYear/df.sum_claimed_lastActivYear).astype('float')
    df['payout_ratio_total'] = np.where(df.sum_claimed_total == 0 ,1,df.sum_payout_total/df.sum_claimed_total).astype('float')
    
    # drop absolute columns
    df = df.drop(columns = ['sum_payout_total','sum_retained_total','sum_payout_lastActivYear','sum_retained_lastActivYear'])
    
    # drop outleiers
    df = df.loc[(df.payout_ratio_total.between(0,1) == True) & (df.payout_ratio_lastActivYear.between(0,1)  == True)]
    
    return df
#_________________________________________________________________________________________________________________________

# main preprocessing function 
def transform_df(contracts, year_only = False, drop_cols = [], claim_ratios = True, cut_effEnd = False, cut_date = '2030-12-31'): 
    '''
    transform preprocessing steps on imported df and return df again.
    
    input:
        • year_only: bool, default=False
            o Set to True to keep only year of all datetime features and convert them to int. Otherwise year and month will be separated and kept.
        • Drop_cols: list of strings, default = []
            o Inserted strings will be tried to be dropped. If col name doesn’t exist or already got dropped within the regular preprocessing process a corresponding message will be printed.
        • claim_ratios: bool, default=True
            o If set to True, some claims related columns will be dropped or replaced and cleaned to minimize correlating columns. In specific: retained columns get dropped, payout amount columns cleaned and replaced by a ratio of claimed amount.
        • cut_effEnd: bool, default=False
            o Set to True to cut policy_effEndDate values at a specific cut_date to tighten distribution.
        • cut_date: datetime, default = ‘2030-12-31’
            o Optional, if cut_effEnd == True. All policy_effEndDate values > cut_date will be replaced by this value.

    return: 
    preprocessed pandas df  
    '''
    print('preprocessing results:')
    
    # get initial number of rows and cols
    init_rows = len(contracts) 
    init_cols = contracts.columns
    
    # set contractID as Index, unique value and should not be used as feature
    if 'ContractID' in contracts.columns:
        contracts.set_index('ContractID', inplace=True)
    
    
    # 1.1 Datatypes
    contracts = transform_dtypes(contracts)
    
    # replace 'Y/N' values by boolean (True/False)
    bool_map = {'Y':True, 'N':False}
    for col in ['expatriate','additional_insurance']: 
        contracts[col] = contracts[col].map(bool_map)
    
    # convert gender column to encoded columns
    if 'insured_Gender' in contracts.columns:
        # Get one hot encoding of columns insured_Gender
        one_hot = pd.get_dummies(contracts['insured_Gender'],prefix='I_gender')
        # Drop column insured_Gender as it is now encoded
        drop_cols.append('insured_Gender')
        # Join the encoded df
        contracts = contracts.join(one_hot)
        contracts.drop(columns=['I_gender_M'],inplace = True) #drop col to keep F & E
    
    # 1.2 drop & replace --> append to input 'drop_cols' and drop cols in the end
    drop_cols.append('MainProductName') #redundant
    
    drop_cols.append('terminationDate') # use only boolean value 'terminated', otherwise NaN values and unwanted info for modelling                  
       
    # replace cols
    
    # calculate timediff instead of date cols
    contracts['policyAge'] = np.round((contracts.RefDate - contracts.policy_startDate) / np.timedelta64(1, 'M'),0)#.astype(int)
    contracts['insured_Age'] = np.round((contracts.update_Date - contracts.insured_birthDate) / np.timedelta64(1, 'Y'),0).astype(int)
    
    # because of high correlation to start date replace applydate by diff to startDate & fillna by 0 & replace outliers <0 by 0
    contracts['ApplyDate'].fillna(contracts.policy_startDate, inplace=True)
    contracts['ApplyDays'] = np.round((contracts.policy_startDate - contracts.ApplyDate) / np.timedelta64(1, 'D'),0).astype(int)
    contracts.loc[contracts['ApplyDays'] < 0, 'ApplyDays'] = 0
    
    # optional: cut policy_effEndDates
    if cut_effEnd:
        # Set values greater than the target date to the target date
        B = len(contracts.loc[contracts['policy_effEndDate'] > cut_date])
        contracts.loc[contracts['policy_effEndDate'] > cut_date, 'policy_effEndDate'] = cut_date
        A = len(contracts.loc[contracts['policy_effEndDate'] > cut_date])
        if B>A:
            print('policy_effEndDate cut off at',cut_date,'for',B-A,'lines.')        
    
    
    # replace paid_until by boolean value, telling if it got paid or not
    contracts['paid'] = np.where(contracts.paid_until.isnull(),0,1) 
    
    #drop now redundant cols (update_Date is uniform and policy_initialEndDate not important, if we have effEndDate)
    for col in ['insured_birthDate','ApplyDate','paid_until','RefDate','SignDate', 'update_Date', 'policy_initialEndDate']:
        drop_cols.append(col) 
        
    # drop all 'lastYear'-cols, since we have lastActiveYear cols
    for col in contracts.columns:
        if 'lastYear' in col:
            drop_cols.append(col)
    
    # drop now all mentioned cols plus additonally chosen columns 'drop_cols' 
    #print(drop_cols) #checkup
    if len(drop_cols)>0:
        for col in drop_cols:
            try:
                contracts.drop(columns=[col], inplace=True)
            except:
                print(f'{col} already dropped or col name might not exist\n')
     
    # 1.3 fill missing values
    
    # fill NaN in countries with XX
    for col in ['insured_nationality','holder_country']:
        try:
            contracts[col].fillna('XX', inplace=True)
        except:
            print('NaNs not filled for',col,'- column already dropped.')
    
    
    # replace NaN in mean_payoutDays by mean of col (0 would give false impression of fast payout)
    # not perfect, because this is no correct information, but leave it for now / we might won't use this col anyways..
    mean_cols = ['mean_payoutDays','mean_payoutDays_lastActivYear'] #create list of cols with mean
    mean = contracts[mean_cols].mean().astype(int) #calculate means on cols
    contracts[mean_cols].fillna(mean, inplace=True) #replace NaN values by mean of col
    
    # replace NaN in sum cols by 0
    sum_cols = ['sum_payout_lastActivYear', 'sum_payout_total']
    contracts[sum_cols].fillna(0, inplace= True)
    
    # fill 'terminationReason' by 'None', if not terminated. Exclude later for modelling (unwanted info, because of correlation to 'terminated' col)
    contracts['terminationReason'].fillna('None', inplace = True)
    
    # 1.4 Drop some rows
    # drop contracts without products 
    B = len(contracts)
    contracts = contracts.loc[contracts.product_code.isna() == False]
    A = len(contracts)
    if B>A:
        print(B-A,'lines dropped due to missing product code')
    
    # drop contracts without premium
    B = len(contracts)
    contracts = contracts.loc[contracts.sum_premium_total.isna() == False]
    A = len(contracts)
    if B>A:
        print(B-A,'lines dropped due to missing premium sum')
    
    # drop outliers 
    # claims related columns < 0
    cash_cols = ['sum_payout_total', 'sum_claimed_total', 'sum_retained_total', 'sum_premium_total','mean_payoutDays', 
           'mean_payoutDays_lastActivYear', 'sum_payout_lastActivYear', 'sum_claimed_lastActivYear', 'sum_retained_lastActivYear', 'sum_premium_lastActivYear']
    for col in cash_cols:
        B = len(contracts)
        contracts = contracts.loc[contracts[col] >= 0]
        A = len(contracts)
        if B>A:
            print(B-A,'lines dropped due to negative value of',col)       
    
    #in payout columns
    B = len(contracts)
    contracts = contracts.loc[(contracts.mean_payoutDays < 300 )&(contracts.mean_payoutDays_lastActivYear < 300)]
    A = len(contracts)
    print(B-A, 'lines dropped due to outlier in  mean_payoutDays')
    
    # in num_claims
    B = len(contracts)
    contracts = contracts.loc[(contracts.num_claims_total < 200 )&(contracts.num_claims_lastActivYear < 100)]
    A = len(contracts)
    print(B-A, 'lines dropped due to outlier in num_claims')
    
    # in claims columns
    B = len(contracts)
    contracts = contracts.loc[(contracts.sum_claimed_total < 30000 )&(contracts.sum_claimed_lastActivYear < 20000)]
    A = len(contracts)
    print(B-A, 'lines dropped due to outlier in claims columns')
    
    # in MainProductCode
    for val in contracts['MainProductCode'].value_counts().index:
        num = len(contracts.loc[contracts['MainProductCode'] == val])
        if num < 5:
            contracts = contracts.loc[contracts['MainProductCode'] != val]
            print(f'{num} rows dropped because MainProductCode {val} contains too little contracts.')
    

    # 1.5 Optional: keep only year info from date cols, e.g. for DecisionTrees: 
    #if year_only == True:
    #    for col in contracts.select_dtypes(include=['datetime64[ns]']).columns:
    #        contracts[col] = contracts[col].dt.year
    #        print('date columns transformed to year only')
    
    # 1.5 create year & optional month of start- & effEndDate
    if 'policy_startDate' in contracts.columns:
        contracts['start_Year'] = contracts['policy_startDate'].dt.year
    if 'policy_effEndDate' in contracts.columns:
        contracts['effEnd_Year'] = contracts['policy_effEndDate'].dt.year
        
    if year_only == False:
        contracts['start_Month'] = contracts['policy_startDate'].dt.month
        contracts['effEnd_Month'] = contracts['policy_startDate'].dt.month
    #contracts.drop(columns=['policy_startDate','policy_effEndDate'], inplace=True) #drop later at train/test-split (to keep dataviz running for preprocessed df too)
        
            
    # create payout ratio(s) & drop absolute cols
    if claim_ratios == True:
        contracts = built_claim_ratios(contracts)
        print('claims ratios created')
    
    # 1.7 create alternative target value
    ds_reasons = ['10014','10015','10016']
    contracts['ds_terminated'] = np.where(((contracts.terminated == 1) & (contracts.terminationReason.isin(ds_reasons))), 1, 0)

    # print final results
    final_rows = len(contracts)
    final_cols = contracts.columns
    new_cols = list(set(final_cols).difference(set(init_cols).difference(drop_cols)))
    dropped_rows = init_rows - final_rows
    dropped_cols = len(drop_cols)
    created_cols = len(new_cols)
    
    if len(drop_cols)>0:
        print('\ndropped columns:',drop_cols)
    print('\ncreated columns:',new_cols)
    print('__________________________________________________________\n')
    print('done. total result:')
    print(f'{dropped_rows} rows dropped. {final_rows} remaining')
    print(f'{dropped_cols} cols dropped. {created_cols} newly created --> {len(final_cols)} cols remaining')
    print('__________________________________________________________\n')
              
    #clear drop_cols
    drop_cols.clear()
    return contracts

#_________________________________________________________________________________________________________________________

def transform_contracts(contracts, year_only = False, drop_cols = [], claim_ratios = True, cut_effEnd = False, cut_date = '2030-12-31', print_terminal=True):
    '''
    uses 'transform_df' to preprocess contracts df and adds option to hide or unhide prints by:
    - print_terminal = True # Options: True, False
    
    info from transform_df:
        transform preprocessing steps on imported df and return df again.

        input:
        • year_only: bool, default=False
            o Set to True to keep only year of all datetime features and convert them to int. Otherwise year and month will be separated and kept.
        • Drop_cols: list of strings, default = []
            o Inserted strings will be tried to be dropped. If col name doesn’t exist or already got dropped within the regular preprocessing process a corresponding message will be printed.
        • claim_ratios: bool, default=True
            o If set to True, some claims related columns will be dropped or replaced and cleaned to minimize correlating columns. In specific: retained columns get dropped, payout amount columns cleaned and replaced by a ratio of claimed amount.
        • cut_effEnd: bool, default=False
            o Set to True to cut policy_effEndDate values at a specific cut_date to tighten distribution.
        • cut_date: datetime, default = ‘2030-12-31’
            o Optional, if cut_effEnd == True. All policy_effEndDate values > cut_date will be replaced by this value.

        return: 
        preprocessed pandas df
    '''
    if print_terminal:
        return transform_df(contracts, year_only = year_only, drop_cols = drop_cols, claim_ratios = claim_ratios, cut_effEnd = cut_effEnd, cut_date = cut_date)
    else:
        with HiddenPrints():
            return transform_df(contracts, year_only = year_only, drop_cols = drop_cols, claim_ratios = claim_ratios, cut_effEnd = cut_effEnd, cut_date = cut_date)

#_________________________________________________________________________________________________________________________

def save_df(df ,filename='contracts_preprocessed'):
    '''
    input: 
    - df 
    - filename (optional, default:'contracts_preprocessed')
    
    saves df to 'preprocessed/' subfolder as str(filename)+'.csv'
    
    example: save_df(contracts)
    '''
    path = 'preprocessed/'+str(filename)+'.csv'
    df.to_csv(path)
    print('df saved to',path)
    
#_________________________________________________________________________________________________________________________

# function to import
def get_products():
    '''
    loads product informations from 'BDAE_DataMining_Products.csv' (must be in same folder)
    '''
    products = pd.read_csv('BDAE_DataMining_Products.csv', encoding = 'ISO-8859-1')
    return products
    
#_________________________________________________________________________________________________________________________

# preprocessing
def transform_products(products, drop_cols = []):
    '''
    preprocess products df
    
    input:
    - products (df)
    - optional: 'drop_cols' = list of column names to drop
    
    return: products 
    '''
        
    # 1.1 Set product_code as index
    products.set_index('product_code', inplace=True)

    # 1.2 Dtypes
    products.replace({'Y':1,'N':0}, inplace=True)
    products = products.convert_dtypes()

    # 1.3 dropna
    # 2 Columns don't have a *MainProductCode*. They seem to be test products and thus can be deleted. 
    products.dropna(subset=['MainProductCode'], inplace=True)
    # 3 columns have NaN product_category --> product_group = Gebühren/Zuschläge (Fees/Surcharges) --> not interesting for contracts --> drop
    products.dropna(subset=['product_category'], inplace=True)
    
    # drop redundant columns
    drop_cols.append('product_groupName')
    drop_cols.append('MainProductName')
    # optional: drop cols
    if len(drop_cols)>0:
        for col in drop_cols:
            try:
                products.drop(columns=[col], inplace=True)
                print(f'{col} column dropped')
            except:
                print(f'{col} column cannot be dropped. column name might not exist.')
    
    return products
    
#_________________________________________________________________________________________________________________________

def merge_products_to_contracts(products, contracts):
    '''
    merge products df to contracts df as left join ON 'product_code' column
    
    input:
    - products (df)
    - contracts (df)
    - optional: 'drop_cols' = list of column names to drop before merge
    
    return:
    contracts, with merged infos from products 
    '''
    
    # merge with contracts
    cols = products.columns.difference(contracts.columns) #avoid column duplicates
    
     # use product_code as key + check again, if unique in product df. + keep ContractID as Index
    contracts = contracts.reset_index().merge(products[cols], how = 'left', on='product_code', validate='m:1').set_index('ContractID')    # convert product_code as str again
    contracts.product_code = contracts.product_code.convert_dtypes()

    #fillna
    max_policyAge_Months = contracts['policyAge'].max() #get in Months
    max_Age = contracts['insured_Age'].max()
    contracts['max_policyDuration(M)'].fillna(max_policyAge_Months, inplace=True)
    contracts['max_renewalDuratio(M)'].fillna(0, inplace=True) #might cause false impression, but try this for the beginning
    contracts['max_renewals'].fillna(0, inplace=True)
    contracts['min_age'].fillna(0, inplace=True)
    contracts['max_age'].fillna(max_Age, inplace=True)
    contracts['max_signAge'].fillna(max_Age, inplace=True)
    
    print('prodduct columns added:',products.columns)

    return contracts
    
#_________________________________________________________________________________________________________________________

# c.1
def country_to_region_mapping(col, iso = True, subregion = False):
    """
    Generates a dictionary mapping country name or ISO codes to their corresponding (sub-)regions.

    Parameters:
        col (df column): containing country names or ISO-Codes. Will be transformed to list
        iso: define, if column contains iso codes or names. Default: True
        subregion: If True subregion (e.g. 'North Europe'), else region (e.g. 'Europe') is mapped. default: False
        

    Returns:
        mapping: Dictionary mapping ISO codes to regions.

    Example Usage:
        col = contracts_encoded.insured_nationality # col containing nation ISO Codes
        iso = True
        subregion = False # use regions
        nationISO_mapping = country_to_region_mapping(col)
        print("Country to Region Mapping:", nationISO_mapping)
        contracts_encoded.insured_nationality.replace(nationISO_mapping) # replace nations by regions
    """
   
    # Create List from country column
    col.fillna('XX', inplace=True) # fill with 'XX'
    countries = col.unique().tolist() #save unique values to list
    
    # empty dict for mapping
    mapping = {}

    # Special feature: 'DE' / 'Germany' has its own region and is not count into 'Europe' region
    if iso == True:
        mapping['DE'] = 'Germany'
    else:
        mapping['Germany'] = 'Germany'

    # Special feature: 'XX' has its own region called 'XX'
    mapping['XX'] = 'XX'

    # Make a request to the REST Countries API
    response = requests.get('https://restcountries.com/v2/all')

    if response.status_code == 200:
        countries_data = response.json()

        # Iterate over the countries data and extract ISO code and region information
        for country_data in countries_data:
            
            # get region or subregion, based on input 'subregion'-value
            if subregion == True:
                region = country_data['subregion']
            else:
                region = country_data['region']
            
            if iso == True:
                iso_code = country_data['alpha2Code']
                if iso_code in countries:
                    if iso_code == 'DE':
                        mapping[iso_code] = 'Germany'
                    elif iso_code == 'XX':
                        mapping[iso_code] = 'XX'
                    else:
                        mapping[iso_code] = region
            
            else:
                name = country_data['name']
                if name in countries:
                    if name == 'Germany':
                        mapping[name] = 'Germany'
                    elif name == 'XX':
                        mapping[name] = 'XX'
                    else:
                        mapping[name] = region

    return mapping

#_________________________________________________________________________________________________________________________

# extra function to encode contracts df
def encode_contracts(df, use_regions= True):
    '''
    returns encoded df by grouping countries & nations in regions & create dummy variables.
    '''    
    # 2.1 regions for nation & country
    # Import function 'country_to_region_mapping' from helpers to create mapping dicts
    if use_regions == True:
        # load country_to_region_mapping, if not already exists
        #if not 'country_to_region_mapping' in locals():
         #   from churn_helpers import country_to_region_mapping 
          #  print('country_to_region_mapping loaded')
            
        if 'insured_nationality' in df.columns:
            try:
                # Create mapping dicts
                nationISO_mapping = country_to_region_mapping(df.insured_nationality)
                # replace nations regions before encoding
                df['insured_nationRegion'] = df.insured_nationality.replace(nationISO_mapping)
                df = df.drop(columns=['insured_nationality'])
            except:
                print('insured_nationality could not be mapped to region - column might already dropped or REST-API not available.')

        if 'holder_country' in df.columns:
            try:
                # Create mapping dicts
                country_mapping = country_to_region_mapping(df.holder_country, iso= False)
                # replace countries by regions before encoding
                df['holder_Region'] = df.holder_country.replace(country_mapping)
                df = df.drop(columns=['holder_country'])
            except:
                print('holder_country could not be mapped to region - column might already dropped or REST-API not available..')

    # 2.2 drop cols with too many values & little imporance
    df.drop(columns=['product_code'], inplace=True)

    # terminationReason: Contains "only" ~20 different values. But this info should not be known for future/ active contracts
    # (if not null, then contract is terminated). So we should drop it (for now)
    if 'terminationReason' in df.columns:
        df.drop(columns=['terminationReason'], inplace=True) 
        
    # set ContractID as index to avoid encoding all ContractID
    if 'ContractID' in df.columns:
        df.set_index('ContractID', inplace=True)

    #create dummies
    contracts_encoded = pd.get_dummies(df, drop_first=True)
    
    #drop cols with too litle infos    
    for col in contracts_encoded.columns:
        if 'insured_nationRegion' in col:
            num = contracts_encoded[col].value_counts()[1]
            if num < len(contracts_encoded)/1000:
                contracts_encoded.loc[contracts_encoded[col] == 1,'insured_nationRegion_XX'] = 1 # #count to XX
                contracts_encoded.drop(columns=[col], inplace=True) #drop col
                # extra loop because of error, when using with HiddenPrints():
                try:
                    print(col.encode('utf-8').decode('cp1252'), 'column dropped because it contains <0.1 percent of rows.')
                except:
                    print('')
        elif 'holder_Region' in col:
            num = contracts_encoded[col].value_counts()[1]
            if num < len(contracts_encoded)/1000:
                contracts_encoded.loc[contracts_encoded[col] == 1,'holder_Region_XX'] = 1 # #count to XX
                contracts_encoded.drop(columns=[col], inplace=True) #drop col
                # extra loop because of error, when using with HiddenPrints():
                try:
                    print(col.encode('utf-8').decode('cp1252'), 'column dropped because it contains <0.1 percent of rows.')
                except:
                    print('')

    return contracts_encoded
    
    
#_________________________________________________________________________________________________________________________

    
# copy to churn_helpers if no changes made
def normalize(X_train, X_test, scale = True, scaler = MinMaxScaler()):
    '''
    normalizes X_train & X_test.
    
    input: X_train, X_test, scale = True, scaler = MinMaxScaler()
    
    return X_train, X_test
    '''
    
    
    if scale and scaler:
        X_cols = X_train.columns

        # should not be necassary anymore, since year & month got seperated
        '''
        #to avoid errors: exclude  datetime columns from scaling, if existant:
        if any(X_train.dtypes == 'datetime64[ns]'):
            noDates = X_train.select_dtypes(exclude='datetime64').columns # select only non-datetime columns to scale
            X_train[noDates] = scaler.fit_transform(X_train[noDates])
            X_test[noDates] = scaler.transform(X_test[noDates])
            print('Datetime columns successfully excluded.')

        else:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        '''
        
        # avoid errors, if data got not encoded:
        if any(X_train.dtypes == 'string'):
            noString = X_train.select_dtypes(exclude='string').columns # select only non-string columns to scale
            X_train[noString] = scaler.fit_transform(X_train[noString])
            X_test[noString] = scaler.transform(X_test[noString])
            print('String columns successfully excluded.')

        else:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        

        #convert np arrays to pandas df to keep col names
        X_train = pd.DataFrame(X_train, columns=X_cols)
        X_test = pd.DataFrame(X_test, columns=X_cols)
        print('Features sucessfully scaled.')
        return X_train, X_test
    
    else:
        print('scale or scaler not set.')
    
    
#_________________________________________________________________________________________________________________________




#_________________________________________________________________________________________________________________________

# Dataviz

#_________________________________________________________________________________________________________________________
 
def show_term_reasons(df):   
    # create figure
    fig = plt.figure(figsize=(18, 10))

    # setting values to rows and column variables
    rows = 1
    columns = 2

    # reading images
    Image2 = cv2.imread('images/ERP-System+SQL/TerminationReasons_En.png')

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # plot donut chart of termination reasons
    data = df.loc[df.terminated == 1]['terminationReason'].value_counts()
    labels = data.index
    plt.pie(data, labels = labels)

    # draw circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()

    # Adding Circle in Pie chart
    fig.gca().add_artist(centre_circle)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title('termination reasons of ended contracts');

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(Image2)
    plt.axis('off')
    plt.title("Termination reasons (English)");

#_________________________________________________________________________________________________________________________

def plot_target_dist(df, target, df_title):
    '''
    input:
    - df (df)
    - tagret (col name) --> "terminated" ( "ds_terminated") 
    
    plots pie chart & values of target value distribution in df.
    '''
    df = df[target]
    
    plt.figure(figsize= (10, 5))
    plt.pie(df.value_counts(),
                        autopct = '%.1f',
                        explode = [0.1,0],
                        labels = df.unique().tolist(),
                        shadow = True, 
                        textprops = {'fontsize':10},
                        colors = ['lightblue', 'steelblue'],
                        startangle = 180
    )
    plt.title(f'Ratio of: {target}-col in {df_title}', fontsize = 12)
    #plt.legend(fontsize = 12, loc = 'upper right')
    plt.show()
    print(df.value_counts())


#_________________________________________________________________________________________________________________________

def timeline_terminated(df, date_col, target='terminated', groupby='Y', ax = None):
    '''
    input:
    - df 
    - date_col (groupby --> x-axis)
    - target = 'terminated'
    - groupby='Y' (options: 'Y', 'M') -_> time period to groupby
    - ax = None (if set: use ax in subplots)

    return None    (plot timeline)
    '''

    # Convert the date column to datetime format if it's not already.
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract the year and month as a new column in 'yyyy-mm' format.
    df['Time'] = df[date_col].dt.to_period(groupby)


    monthly_ratio_data = df.groupby('Time')[target].mean().reset_index()
    monthly_ratio_data.Time = monthly_ratio_data.Time.astype(str)


    # Plot the data using seaborn
    if ax:
        sns.lineplot(data=monthly_ratio_data, x='Time', y=target, ax=ax)

        # Customize the subplot
        ax.set_title("Monthly Ratio of '{}' Values Over Time".format(target))
        ax.set_xlabel(date_col)
        ax.set_ylabel(f"Ratio of {target} Values")
        ax.grid(True)
        ax.tick_params(axis='x', labelrotation = 45) # Rotating X-axis labels

    else:
        # Create a plot
        plt.figure(figsize=(10, 5))

        sns.lineplot(data=monthly_ratio_data, x='Time', y=target)

        # Customize the plot
        plt.title("Monthly Ratio of '{}' Values Over Time".format(target))
        plt.xlabel(date_col)
        plt.ylabel("Ratio of '{}' Values".format(target))
        plt.grid(True)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

        # Display the plot
        plt.show();

    df.drop(columns=['Time'], inplace=True) #drop col again

    return 

#_________________________________________________________________________________________________________________________

def plot_upper_corr(df, t = 0.9):
    '''
    plot upper triangle heatmap of absolute correlations from input df.
    
    input:
    - df
    - t = 0.9 --> treshhold: plot only corrs > t
    '''
    # calculate absolute correlations
    corrs = df.corr().abs()

    # create upper triangle & drop diag (with corr = 1)
    upper = corrs.where(np.triu(np.ones(corrs.shape),k=1).astype(bool))

    # select correlations higher than treshold t
    upper_t = upper.loc[upper.max(axis=1) > t, upper.max(axis=0) > t]

    # plot heatmat of correlations
    plt.figure(figsize=(10, 5))
    heatmap = sns.heatmap(upper_t, vmin=0, vmax=1, annot=True, cmap='coolwarm')
    heatmap.set_title(f'Heatmap of |Correlations| > {t}', fontdict={'fontsize':10}, pad=12);

#_________________________________________________________________________________________________________________________

def plot_target_corr(df, target_col = 'terminated', k = 20, encode = True, fs = 20):
    '''
    plots top k columns due to absolute correlation with target column.
    
    input:
    - df
    - target_col = 'terminated'
    - k = 20 (if set, show only top k correlated columns. If all corrs should be plotted: set to None)
    - encode = True (encode to show corrs with all columns)
    - fs = 20 (defines fontsize of labels, values & title)
    '''
    if encode:
        with HiddenPrints(): #Hide output
            df = encode_contracts(df) #encode to show corrs with all columns
    corrs = df.corr()[target_col]
    corrs = corrs[corrs.index != target_col]  # Exclude the target column itself
    abs_corrs = corrs.abs().sort_values(ascending=False)
    if k:
        abs_corrs = abs_corrs.head(k) # use absolute correlations to compare pos and neg corrs
        plt.figure(figsize=(15, k/2)) # set figure size depending on selected amount of features
    
    else:
        n = len(df.columns) # else: fit to number of columns
        plt.figure(figsize=(15, n/2))
    #use absolute correlations for better sorting but highlight if the corr is negativ or positiv
    sns.barplot(x=abs_corrs.values, y=abs_corrs.index, palette=["g" if corr > 0 else "r" for corr in corrs])
    plt.xlabel('Absolute Correlation', fontsize=fs)
    plt.ylabel('Column Name', fontsize=fs)
    sns.set(rc={'xtick.labelsize': fs, 'ytick.labelsize': fs})
    if k:
        plt.title(f'Top {k} Absolute Correlation with {target_col}-Column (Green = Positive, Red = Negative)', fontsize=fs+2)
    else:
        plt.title(f'Absolute Correlation with {target_col}-Column (Green = Positive, Red = Negative)', fontsize=fs+2)
    
    plt.show();

#_________________________________________________________________________________________________________________________

def plot_outliers(df, dtype = 'number', top_k = 4, plot='box'):
    '''
    plot scatterplots for columns with most rel. difference between min and max:
    
    input:
    - df
    - dtype = 'number' (options: 'number', 'float', 'int')
    - top_k = 4 (number of columns to plot scatterplots for)
    - plot = 'scatter' (options: 'scatter', 'box') --> kind of plot for outliers
    '''    
    cols = df.select_dtypes(include=[dtype]).columns
    
    rels = {}
    for col in cols:
        Cmax = df[col].max()
        Cmin = df[col].min()
        Cmean = df[col].mean()
        Crel = np.abs(Cmax - Cmin)/Cmean
        rels[col] = Crel.round(1)        
    # sort by values
    rels = dict(sorted(rels.items(), key=lambda item: item[1]))
    print(f'oredered ratio of max to min value in {dtype} cols:\n', rels)

    outlier_cols = list(rels)[-top_k:]
    print(f'\n top {top_k} outlier_cols of dtype {dtype} due to min-max-ratio:\n',outlier_cols)
          
    if plot == 'scatter':
        g = sns.PairGrid(data=df,vars=outleier_cols, hue="terminated")
        g.map(sns.scatterplot)
        g.add_legend();
          
    elif plot in ['violin','box']:
        # Calculate the number of rows and columns in the subplot grid
        num_cols = 2
        num_rows = math.ceil(len(outlier_cols) / num_cols)

        # Create a grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        fig.suptitle(f'{plot}-plots for top {top_k} outliers of dtype {dtype}', fontsize=16)

        # Flatten the axes array if it's multidimensional
        axes = axes.flatten()

        # Create violin plots for each specified column
        for i, col in enumerate(outlier_cols):
            if plot == 'violin':
                sns.violinplot(x=df[col], ax=axes[i], inner="quartile")
            elif plot == 'box':
                sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(col)

        # Hide any empty subplots
        for i in range(len(outlier_cols), num_rows * num_cols):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()
    return

#_________________________________________________________________________________________________________________________


#_________________________________________________________________________________________________________________________


#_________________________________________________________________________________________________________________________


#_________________________________________________________________________________________________________________________

# Modelling

#_________________________________________________________________________________________________________________________
 
def create_train_test(df, 
                      test_size=0.2, 
                      ds_target = False, 
                      ds_reasons = ['10014','10015','10016'], 
                      encoder = 'CountFrequency',
                      print_shape = False,
                      feature_selection = [],
                      split_by_date = False,
                      split_date_col = 'policy_startDate',
                      test_nan = 'drop'):
    '''
    create encoded train & test data from df
    
    input:
    - df
    - test_size = 0.2 (ratio of test data)
    - ds_target = False (if True --> target value y is set to ds_terminated)
    - ds_reasons = ['10014','10015','10016'] (termination reasons that are used for creation of ds_terminated)
    - encoder = CountFrequency (encoder for categorical variables, other option: 'getDummies', None [if already encoded])
    - feature_selection = [] (insert col-names for selection, otherwise all columns will be taken into account)
    - test_nan = 'drop' (handling of NaN values in test set after encoding. other option: 'fill')
    
    return:
    - X_train, X_test, y_train, y_test 
    - encoder (only if encoder is set to 'CountFrequency') --> to reverse transform X later
    '''
    # (re-)set ContractID as Index (if loaded from csv file)
    if 'ContractID' in df.columns:
        df.set_index('ContractID', inplace=True)

    # transform dtypes (if lost by csv import) 
    if 'object' in df.dtypes.values:
        df = transform_dtypes(df)
    
    #(re-)create alternative target value
    if 'ds_terminated' not in df.columns:
        df['ds_terminated'] = np.where(((df.terminated == 1) & (df.terminationReason.isin(ds_reasons))), 1, 0)
    
    term_cols = ['terminated','ds_terminated','terminationReason']
    # optional: feature selection
    if len(feature_selection) > 0:
        df = df[feature_selection + term_cols]
    
        
    if encoder == 'getDummies':
        df = encode_contracts(df)
        term_cols.remove('terminationReason') #already dropped before dummy creation
            
    #optional: split by date:
    if split_by_date == True:
        df = df.sort_values(split_date_col, ascending = True)
        df.drop(columns=['policy_startDate','policy_effEndDate'], inplace=True) # drop columns now to keep only year (& month)
        train_set, test_set= np.split(df, [int((1-test_size) *len(df))])
        X_train = train_set.drop(columns=term_cols)
        X_test = test_set.drop(columns=term_cols)
        if ds_target == True:
            y_train = train_set.ds_terminated
            y_test = test_set.ds_terminated
        else:
            y_train = train_set.terminated
            y_test = test_set.terminated
    else:
        df.drop(columns=['policy_startDate','policy_effEndDate'], inplace=True) # drop columns now to keep only year (& month)
        if ds_target == True:
            y = df.ds_terminated

        else:
            y = df.terminated

        # drop target (related) variables & create X
        X = df.drop(columns=term_cols)
    
    # encode
    if encoder == 'CountFrequency':
        # collect variables & set to categorical
        cat_cols = list(df.drop(columns=term_cols).select_dtypes(include=['string','category']).columns)
        
        #split only if not already splitted by date
        if split_by_date == False:
            for col in cat_cols:
                X[col] = pd.Categorical(X[col])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
        else:
            for col in cat_cols:
                X_train[col] = pd.Categorical(X_train[col])
                X_test[col] = pd.Categorical(X_test[col])
        
        
        # set up the encoder
        encoder = CountFrequencyEncoder(encoding_method='frequency', variables=cat_cols) # alternative: encoding_method='count'
                                       
        # fit the encoder
        encoder.fit(X_train)

        # transform the data
        train_t= encoder.transform(X_train)
        test_t= encoder.transform(X_test)
        
        #drop rows in test set with NAN-values (because encoded col value only exists in X_train)
        for col in ['holder_country','insured_nationality']:
            if col in cat_cols:
                if test_nan == 'drop':
                    y_test = y_test.loc[test_t[col].isna() == False]
                    test_t = test_t.loc[test_t[col].isna() == False]
                elif test_nan == 'fill':
                    test_t[col].fillna(0, inplace=True)
        
        # convert y_train & y_test to arrays        
        y_train, y_test = np.array(y_train.astype('int')), np.array(y_test.astype('int'))
        
        #print(train_t.head())
        enc_dict = encoder.encoder_dict_
        
        if print_shape == True:
            print(f'{train_t.shape = } \n{test_t.shape = } \n{y_train.shape = } \n{y_test.shape = }') #test
        
        print(f'{encoder = }')
        return train_t, test_t, y_train, y_test, encoder
    
    # if not encoded..            
    #split only if not already splitted by date
    if split_by_date == False:   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
    # check, that y are arrays
    y_train, y_test = np.array(y_train.astype('int')), np.array(y_test.astype('int'))
    
    if print_shape == True:
        print(f'{X_train.shape = } \n{X_test.shape = } \n{y_train.shape = } \n{y_test.shape = }') #test

    return X_train, X_test, y_train, y_test    
    
#_________________________________________________________________________________________________________________________


#_________________________________________________________________________________________________________________________

       
#_________________________________________________________________________________________________________________________

 # copy to churn_helpers and remove, if not changed anymore
def param_summary(prepro_params = prepro_params, split_params = split_params, scaling_params = scaling_params):
    '''
    print all selected parameters
    '''
    if prepro_params:
        print('selected preprocessing parameters:')
        for param_name, param_value in prepro_params.items():
            print(f'{param_name} = {param_value}')
        print('_________________________________________________________\n')
    if split_params:
        print('selected train-/test-split parameters:')
        for param_name, param_value in split_params.items():
            print(f'{param_name} = {param_value}')
        print('_________________________________________________________\n')
    if scaling_params:
        print('selected scaling parameters:')
        for param_name, param_value in scaling_params.items():
            print(f'{param_name} = {param_value}')
        print('_________________________________________________________\n')
    
#_________________________________________________________________________________________________________________________

# copy to churn_helpers and remove, if not changed anymore
def model_summary(model , y_train = y_train, y_test = y_test, ds_target = ds_target, filename = filename, scale=scale, scaler = scaler):
    
    #print model
    print('selected model:',model.__class__.__name__)
    
    # print target
    if ds_target:
        print('selected target: ds_terminated with ds_reasons =', ds_reasons)
    else:
        print('selected target: terminated')
    print('selected preprocessing file:',filename)
    
    #print encoder
    try:
        print('Encoded by:',encoder.__class__.__name__)
    except:
        print('Encoded by: GetDummies')
    
    # print normalizer
    if scale:
        print('Normalized by',scaler.__class__.__name__)
    
    # print split-type
    if split_by_date:
        print('train & test data splitted by date')
    else:
        print('train & test data splitted randomly')
    
    # print target ratio
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    print(f'ratio of positive class within train data: {train_ratio:.1%} and test data {test_ratio:.1%}')
    print('_________________________________________________________\n')


#_________________________________________________________________________________________________________________________

# copy to churn_helpers and remove, if not changed anymore
def eval_model(model, X_train , X_test , y_train = y_train, y_test = y_test, data= 'all',plot_CM = True, norm_CM = 'true',model_infos = True, ds_target = ds_target, filename = filename, scale=scale, scaler = scaler, split_by_date = split_by_date):
    '''
    Creates confusion matrix and classification reports for input model and returns predictions for X_train & X_test.
    
    input:
    - model
    - X_train
    - X_test
    - data: 'train' / 'test' / 'all' (default: 'all') --> decides for which X data classification reports will be printed
    - plot_CM = True (True / False) --> plot Confusion Matrix
    - normalize{‘true’, ‘pred’, ‘all’, None}, default='true'
        -> Either to normalize the counts display in the matrix:
            if 'true', the confusion matrix is normalized over the true conditions (e.g. rows);
            if 'pred', the confusion matrix is normalized over the predicted conditions (e.g. columns);
            if 'all', the confusion matrix is normalized by the total number of samples;
            if None, the confusion matrix will not be normalized.
    - model_infos = True --> decide to plot infos about data prepro & model
        
    print:
    - selected parameters for preprocessing, train-test-split and scaling
    - classification report
    - confustion matrix
    
    return: y_pred_train, y_pred_test
    '''
    #summary of selected options:
    if model_infos:
        model_summary(model, y_train, y_test)
    
    # create predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    if data == 'train':
        print('Classification Report for train data:\n\n', classification_report(y_train, y_pred))
        if plot_CM:
            ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, normalize=norm_CM)
            plt.title(f'Confusion Matrix of Train Data. Normalized by: {norm_CM}')
    elif data == 'test':
        y_pred = model.predict(X_test)
        print('Classification Report for test data:\n\n', classification_report(y_test, y_pred))
        print('F1 score on test set:', f1_score(y_test, y_pred_test).round(2))
        if plot_CM:
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize=norm_CM)
            plt.title(f'Confusion Matrix of Test Data. Normalized by: {norm_CM}')
    elif data == 'all':
        print('Classification Report for train data:\n\n',classification_report(y_train, y_pred_train))
        print('_________________________________________________________\n')
        print('Classification Report for test data:\n\n', classification_report(y_test, y_pred_test))
        print('F1 score on test set:', f1_score(y_test, y_pred_test).round(2))

        if plot_CM:
            fig, axes = plt.subplots(1,2, sharey=True)
            plt.suptitle(f'Confusion Matrix. Normalized by: {norm_CM}')
            ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, normalize=norm_CM, ax=axes[0])
            axes[0].set_title(f'Train Data')
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize=norm_CM, ax = axes[1])
            axes[1].set_title(f'Test Data')
    else:
        print('data value not set correctly')
    return y_pred_train, y_pred_test
       

#_________________________________________________________________________________________________________________________
    
# copy to churn_helpers and remove, if not changed anymore
def shap_create_and_summary(model, data = X_train, plot = 'summary', example = None, model_infos = True,y_train = y_train, y_test = y_test, scale=scale, scaler = scaler, split_by_date = split_by_date,Explainer = 'Explainer'):
    '''
    creates shap explainer and plots summary
    
    input:
    - model
    - data: X_train, X_test (default = X_train)
    - plot: kind of plot ('summary' / 'bar' / 'beeswarm', default = 'summary')
    - example: plot example for this sample (default: None)
    - model_infos: decide to plot infos about data prepro & model (default = True)
    
    return:
    explainer, shap_values
    
    example: 
    xgb_explainer, xgb_shap_values = shap_create_and_summary(xgb_model)
    '''
    # Get Shap Values
    
    # initiate explainer
    if Explainer == 'Explainer':
        explainer = shap.Explainer(model=model)
    elif Explainer == 'TreeExplainer':
        explainer = shap.TreeExplainer(model=model) 

    shap_values = explainer(data)
    
    #reshape shap values, if they are 3dim
    if len(shap_values.shape) == 3:
        shap_values_3d = shap_values
        # Extract the first column of SHAP values
        shap_values_first_col = shap_values_3d.values[:, :, 0]
        base_values_first_col = shap_values_3d.base_values[:,0]
        shap_values = shap.Explanation(shap_values_first_col, base_values=base_values_first_col, data=data)
    
    #summary of selected options:
    if model_infos:
        model_summary(model, y_train, y_test)  
    
    if plot == 'summary':
        # Summary plot
        shap.summary_plot(shap_values, data)
    elif plot == 'bar':
        #most important features
        shap.plots.bar(shap_values)
    elif plot == 'beeswarm':
        shap.plots.beeswarm(shap_values)
    
    # print example waterfall
    if example in range(len(data)):
        print(f'waterfall for sample {example}:\n')
        shap.plots.waterfall(shap_values[example])
    
    return explainer, shap_values


#_________________________________________________________________________________________________________________________

#Function for uniformly results
def get_F1test_and_time(model, TrainTestData):
    '''
    get & save f1-score on test & time to fit & predict 
    
    input: model, TrainTestData = (X_train, X_test, y_train, y_test)
    
    return: f1_test, elapsed_time
    '''
    
    # Unzip TrainTestData
    X_train, X_test, y_train, y_test = TrainTestData
    
    start_time = time.time() # Start the timer

    # Fit the best model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set using the best model
    y_pred = model.predict(X_test)

    # Calculate the F1 score
    f1_test = f1_score(y_test, y_pred).round(2)

    end_time = time.time() # Stop the timer
    elapsed_time = np.round((end_time - start_time), 2)
    print(f'Results of best {model.__class__.__name__} model on train data: F1 Score on Test Data = {f1_test} - Time for fit & predict: {elapsed_time} s')
    
    return f1_test, elapsed_time


#_________________________________________________________________________________________________________________________

#Function2 for uniformly results
def get_F1_and_time(model, TrainTestData):
    '''
    get & save f1-score on train & test & time to fit & predict 
    
    input: model, TrainTestData = (X_train, X_test, y_train, y_test)
    
    return: f1_test, elapsed_time    return f1_train, f1_test, elapsed_time    
    
    '''
    
    # Unzip TrainTestData
    X_train, X_test, y_train, y_test = TrainTestData
    
    start_time = time.time() # Start the timer

    # Fit the best model to the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the train set using the best model
    y_pred = model.predict(X_train)

    # Calculate the F1 score
    f1_train = f1_score(y_train, y_pred).round(2)

    # Make predictions on the test set using the best model
    y_pred = model.predict(X_test)

    # Calculate the F1 score
    f1_test = f1_score(y_test, y_pred).round(2)

    end_time = time.time() # Stop the timer
    elapsed_time = np.round((end_time - start_time), 2)
    print(f'Results of best {model.__class__.__name__} model on train data:')
    print(f'F1 Score on Train Data = {f1_train}')
    print(f'F1 Score on Test Data = {f1_test}')
    print(f'Time for fit & predict: {elapsed_time} s')
    
    return f1_train, f1_test, elapsed_time



#_________________________________________________________________________________________________________________________

def resample_df(df, target = 'ds_terminated', reduce_ratio = reduce_ratio, sample = 'over', plot_dist = True):
    '''
    resample df, plot old and new target distribution and return resampled df
    
    input: 
    df = contracts 
    target = 'ds_terminated' (target variable)
    reduce_ratio = 2 (reduce count of class 0 by only this factor of the difference. for 3 --> difference is afterwards reduced by 1/3)
    sample = 'over' (options: 'over', 'under')
    plot_dist = True (If set to False distributions before and after wont get plotted)
    
    plot: counts of target variable before and after resampling
    
    return: df after resampling
    '''

    # Class count
    count_class_0, count_class_1 = df[target].value_counts()

    # reduce count of class 0 by only factor n of the difference
    reduced_class_0 = int(count_class_1 + ((count_class_0 - count_class_1)/reduce_ratio))

    # Divide by class
    df_class_0 = df[df[target] == 0]
    df_class_1 = df[df[target] == 1]

    if sample == 'under':
        df_class_0_under = df_class_0.sample(reduced_class_0)
        df_new = pd.concat([df_class_0_under, df_class_1], axis=0)
        decr = len(df_class_0_under) / len(df_class_0)
        print(f'undersampled class 0 from {len(df_class_0)} to {len(df_class_0_under)}')
        print(f'class 0 got undersampled by factor {int(decr)} to reduce imbalance within df by {1/reduce_ratio:.0%}')
    elif sample == 'over':
        df_class_1_over = df_class_1.sample(reduced_class_0, replace=True)
        df_new = pd.concat([df_class_0, df_class_1_over], axis=0)
        incr = len(df_class_1_over) / len(df_class_1)
        print(f'oversampled class 1 from {len(df_class_1)} to {len(df_class_1_over)}')
        print(f'class 1 got oversampled by factor {int(incr)} to reduce imbalance within df by {1/reduce_ratio:.0%}')
    
    if plot_dist:    
        # Plot subplots
        fig, (ax1,ax2) = plt.subplots(2,1, figsize = (15, 5), sharex=True)
        plt.suptitle(f'Random {sample}sampling by ratio {reduce_ratio}')

        # plot old distribution
        df[target].value_counts().plot(kind='barh', title='Count before resampling',  ax=ax1)
        ax1.set_ylabel(target)

        # plot new distribution
        df_new[target].value_counts().plot(kind='barh', title='Count after resampling',  ax=ax2)
        ax2.set_ylabel(target);
           
    return df_new
       
#_________________________________________________________________________________________________________________________

def plot_scores(f1_scores = {}):
    # Extract keys and values
    lists = sorted(f1_scores.items()) 
    x, y = zip(*lists)

    # Create a 2D line plot
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title('F1-Scores on test data depending on the reduction of imbalance')
    plt.xlabel('Factor of Reduction of imbalance between class 1 and 0 (1=100%)')
    plt.ylabel('F1-Score on Test data')

    # Find the maximum F1 score and its corresponding x-coordinate
    max_f1_score = max(y)
    max_x = x[y.index(max_f1_score)]

    # Add a red dotted vertical line at the maximum
    plt.axvline(x=max_x, color='red', linestyle='--')

    # Add a red dotted horizontal line at the maximum
    plt.axhline(y=max_f1_score, color='red', linestyle='--')

    # Annotate the maximum F1 score on the plot (middle right, slightly lower)
    plt.annotate(f'Max F1 Score: {max_f1_score:.2f}', xy=(max_x, max_f1_score), xytext=(max_x + 0.5, max_f1_score - 0.02),
                 arrowprops=dict(arrowstyle="->", color='black'))

    # Annotate the maximum imbalance reduction on the x-axis
    plt.annotate(f'Max Imbalance Reduction: {max_x}', xy=(max_x, max_f1_score), xytext=(max_x, max_f1_score - 0.2),
                 arrowprops=dict(arrowstyle="->", color='black'))

    # Display the plot
    plt.show()
    
    return max_x, max_f1_score


#_________________________________________________________________________________________________________________________

# copy to churn_helpers and remove, if not changed anymore
def select_top_k_features(model, 
                          X_train, 
                          X_test,
                          y_train = y_train, 
                          y_test = y_test,
                          k=5,
                         Explainer = 'Explainer'):
    '''
    Uses shap to select top k features of train data based on a selected model and returns only these features in X_train, X_test.
    
    input:
    - model  
    - X_train (df)
    - X_test (df) 
    - k=5 (int, number of top features to be selected, must be in range(len(X_train.columns)))
    - Explainer = 'Explainer' (Kind of shap explainer. Options: 'Explainer', 'TreeExplainer')
    - y_train = y_train
    - y_test = y_test,
    
    return: 
    X_train, X_test
    '''
    #select top k features, if set correctly
    if k in range(len(X_train.columns)):
        #train model
        model.fit(X_train, y_train)
        
        # initiate explainer
        if Explainer == 'Explainer':
            explainer = shap.Explainer(model=model)
        elif Explainer == 'TreeExplainer':
            explainer = shap.TreeExplainer(model=model) 
        
        shap_values = explainer(X_train)
        
        # Calculate feature importances using Shapley values
        feature_importances = np.abs(shap_values.values).mean(axis=0)

        # Sort features based on importance scores
        sorted_features = X_train.columns[np.argsort(feature_importances)[::-1]]

        # Select the top-k features
        selected_features = sorted_features[:k].tolist()
        
        X_train_k = X_train[selected_features]
        X_test_k = X_test[selected_features]
        
        print(f'X_train & X_test got reduced to main {k} features:\n{selected_features}')
        print('_________________________________________________________\n')
        
        return X_train_k, X_test_k
        
    else:
        print('k out of range')
        return
       
#_________________________________________________________________________________________________________________________

# copy to churn_helpers and remove, if not changed anymore
def run_eval_model(model,
                   X_train,
                   X_test,
                   y_train,
                   y_test,                   
                   eval_data = 'all',
                  shap_data = 'all',
                  shap_plot = 'bar',
                  shap_example = None,
                  model_infos = True,
                  Explainer = 'Explainer'):
    '''
    create, train, evaluate & plot shap summary for selected model
    
    input:
    - model ('xgboost', 'DecisionTree')
    - X_train, X_test, y_train, y_test,
    - eval_data = 'all' (other options: 'train', 'test', None) 
    - shap_data = 'all' (other options: 'train', 'test')
    - shap_plot = 'bar' (other options: 'summary', 'beeswarm')
    - shap_example = None (other options: int in range(len(X_test)))
    - model infos = True --> plots infos about prepro & model
    #- sel_top_k_features = None (if set to k in range(len(contracts.columns)) only top k shap features of train data will be used)
    '''
        
    #train model
    model.fit(X_train, y_train)
    
    #evaluate
    if eval_data:
        eval_model(model=model, X_train = X_train, X_test = X_test, y_train= y_train, y_test=y_test, data= eval_data, plot_CM = False, model_infos = model_infos)
    
    #shap data 
    if shap_data in ['train', 'all']:
        print('shap values for X_train:')
        explainer_train, shap_values_train = shap_create_and_summary(model, X_train, plot = shap_plot, example=shap_example, model_infos=False,y_train = y_train, y_test = y_test, Explainer = Explainer)
    
    if shap_data in ['test', 'all']:
        print('shap values for X_test:')    
        explainer_test, shap_values_test = shap_create_and_summary(model, X_test, plot = shap_plot, example=shap_example, model_infos=False,y_train = y_train, y_test = y_test, Explainer = Explainer)
    return model

#_________________________________________________________________________________________________________________________
# Create a function to update the waterfall plot based on the selected value of n
def update_waterfall_plot(n, shap_values = shap_values):
    '''
    plots waterfall for sample n
    '''
    print(f'Waterfall for sample {n}:\n')
    shap.plots.waterfall(shap_values[n])
      
    
#_________________________________________________________________________________________________________________________

def predict_top_n_terminated_contracts(model, input_df, n=10, active_only=False):
    """
    Predict the top n terminated contracts based on their termination probability.

    Args:
    - model: Trained XGBoost classifier.
    - input_df: DataFrame containing contract data and an 'activ' column (0 or 1).
    - n: Number of top contracts to return.
    - active_only: If True, only consider active contracts (activ=1).

    Returns:
    - DataFrame with ContractID and TerminationProbability for the top n terminated contracts.
    - n rows of input df with highest probability (and dropped activ col)
    
    Example usage:
    top_n_terminated, X_test_k_top_n = predict_top_n_terminated_contracts(model = xgb_top10, input_df = X_test_k, n=100, active_only=True)
    """

    # Filter active contracts if active_only is True
    if active_only:
        input_df = input_df[input_df['activ'] == 1]

    # Extract features from the input data
    features = input_df.drop(['activ'], axis=1)

    # Get class probabilities using the XGBoost model
    class_probabilities = model.predict_proba(features)  # This returns probabilities for both classes.

    # Select the probabilities for class 1 (terminated)
    termination_probabilities = class_probabilities[:, 1].round(3)

    # Create a DataFrame with ContractID and TerminationProbability
    result_df = pd.DataFrame({
        'ContractID': input_df.index,
        'TerminationProbability': termination_probabilities
    })
    
    # Set ContractID as index
    result_df.set_index('ContractID', inplace=True)

    # Sort the DataFrame by termination probability in descending order
    result_df = result_df.sort_values(by='TerminationProbability', ascending=False)

    # Select the top n terminated contracts
    top_n_terminated_contracts = result_df.head(n)
    
    # Select top n terminated contracts of input df
    features_top_n = features.loc[top_n_terminated_contracts.index]

    return top_n_terminated_contracts, features_top_n

#_________________________________________________________________________________________________________________________

def predict_probabilities(model, X_train, X_test):
    """
    Predict probabilities for X_train and X_test using the specified model.

    Args:
    - model: Trained XGBoost classifier.
    - X_train: DataFrame containing features for the training data.
    - X_test: DataFrame containing features for the testing data.

    Returns:
    - Tuple of two DataFrames:
      1. DataFrame with probabilities for X_train.
      2. DataFrame with probabilities for X_test.
    """
    
    # Make predictions for the training data
    train_probs = model.predict_proba(X_train)  # This returns probabilities for both classes.
    
    # Make predictions for the testing data
    test_probs = model.predict_proba(X_test)  # This returns probabilities for both classes.
    
    # Create DataFrames with probability columns
    train_prob_df = pd.DataFrame(train_probs, columns=['Probability_Class0', 'Probability_Class1'], index=X_train.index)
    test_prob_df = pd.DataFrame(test_probs, columns=['Probability_Class0', 'Probability_Class1'], index=X_test.index)
    
    return train_prob_df, test_prob_df


#_________________________________________________________________________________________________________________________




#_________________________________________________________________________________________________________________________




#_________________________________________________________________________________________________________________________

def run_grid_search(model, param_grid, TrainTestData, scoring = 'f1', cv = 3, model_name = None):
    '''
    runs grid search and prints & returns results.
    
    input: 
    - model (classifier)
    - param_grid (Dict) 
    - TrainTestData ( = (X_train, X_test, y_train, y_test))
    - scoring = 'f1' 
    - model_name = None (String, optional to name variables saved by joblib, if None model.__class__.__name__ is set)
    
    return best_model, grid_params, f1_grid, f1_train, f1_test, grid_time
    '''
    #extract TrainTestData
    (X_train, X_test, y_train, y_test) = TrainTestData
    
    # Record the start time
    start_time = time.time()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,  # Sf1
        cv=cv,  # 3
        verbose=2,
        n_jobs=-1  # Use all available CPU cores
    )

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Best score
    f1_grid = grid_search.best_score_.round(2)

    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Best Params
    grid_params = grid_search.best_params_

    # Fit the best model to the training data
    best_model.fit(X_train, y_train)

    # Make predictions on the train set using the best model
    y_pred = best_model.predict(X_train)

    # Calculate the F1 score
    f1_train = f1_score(y_train, y_pred).round(2)

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)

    # Calculate the F1 score
    f1_test = f1_score(y_test, y_pred).round(2)
    
    # Record the end time
    end_time = time.time()
    # Calculate the runtime
    grid_time = np.round((end_time - start_time), 3)
    
    # model name
    if not model_name:
        model_name = model.__class__.__name__
    
    # Print results
    print(f'Results for {model_name}:')
    print("Best hyperparameters in GridSearch:",  grid_params)
    print("Best F1 Score in GridSearch:", f1_grid)
    print(f'F1 Score on Train Data = {f1_train}')
    print(f'F1 Score on Test Data = {f1_test}')
    print("---run time for GridSearch, fit & predict = %s seconds ---" % grid_time)

    # Save each variable with a specific filename
    save_variable(best_model, 'best_model_' + model_name)
    save_variable(grid_params, 'grid_params_' + model_name)
    save_variable(f1_grid, 'f1_grid_' + model_name)
    save_variable(f1_train, 'f1_train_' + model_name)
    save_variable(f1_test, 'f1_test_' + model_name)
    save_variable(grid_time, 'grid_time_' + model_name)

    return best_model, grid_params, f1_grid, f1_train, f1_test, grid_time
       
#_________________________________________________________________________________________________________________________

# helpers to save & load variables

#_________________________________________________________________________________________________________________________

def save_variable(variable, filename, subfolder='variables/'):
    '''
    save variable as .pkl file in subfolder
    
    input: variable, filename, subfolder='variables/'
    
    return: None
    '''
    file_path = f'{subfolder}{filename}.pkl'
    joblib.dump(variable, file_path)
    print(f'{filename}.pkl saved to {subfolder}')       
    
#_________________________________________________________________________________________________________________________
def load_variable(var_name, subfolder='variables/'):
    '''
    load variable, if its var_name exists as .pkl file in subfolder.
    
    input: var_name (string)
    
    return: var
    '''
    # Define the file path where the variables should be saved
    filename = subfolder + var_name + '.pkl'

    # if files exist: load
    if not os.path.exists(filename):
        print(filename, 'not found.')
    else:
        # Load the pre-trained variables if it already exists
        var = joblib.load(filename)
        print(filename, 'loaded.')
        return var 
       
#_________________________________________________________________________________________________________________________

# Streamlit helpers
#_________________________________________________________________________________________________________________________

class StreamlitPrintOutput:
    def __init__(self):
        self.output = io.StringIO()
        self._stdout = sys.stdout  # Store the original stdout

    @contextlib.contextmanager
    def capture_print_output(self):
        sys.stdout = self.output
        yield self

    def display_output(self):
        self.output.seek(0)
        st.text(self.output.read())
        self.output.truncate(0)  # Clear the captured output
        sys.stdout = self._stdout  # Restore the original stdout

#print_output = StreamlitPrintOutput()
