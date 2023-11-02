# Churn Predictions of international health insurance contracts

## Introduction
The goal of this sub-project is to predict if (/probability that) a contract will be terminated by the customer. In particular:
1. Which contrcats have the highest probability to get terminated by the customer?
2. What are main global impacts on contract terminations?
3. What are individual factors for customers to terminate their contracts?

Therefore different kinds of contract related information were extracted from the ERP System SAP B1 and processed in SQL to create a pseudonimised csv file with features per contract. The most recent version is __[BDAE_DataMining_Policies_v2.csv](/ChurnProject/BDAE_DataMining_Policies_v2.csv)__.
![Data Collection Steps](https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/assets/134729968/bd8a03c6-9d32-47e8-af7d-ceb3ba21588a)

## Processing
The data got then further processed in 2 Jupyter Notebooks:
1. __[Churn_Introduction+DataExploration](/ChurnProject/Churn_Introduction%2BDataExploration.ipynb)__:
  - Introduction, first Overview, Data Exploration and first Modelling attempt to get main features with DecisionTree.

2. __[Churn_Preprocessing+Modelling](/ChurnProject/Churn_Preprocessing%2BModelling.ipynb)__:
  - Data Cleaning & Preprocessing, Data Visualisation & further exploration of correlations, cistributions & Co.
  - Train- & Test-Data creation, including alternative target variable for specific termination reasons only.
  - Classification predictions with XGBoost-, SupportVector- & RandomForest-Classifier on both target variables. Tuning attempts, comparison & interpretation with SHAP.
  -  Probability predictions with XGBoost, interpretation with SHAP.

## Results
Main preprocessing steps, data visualisations and modeelling results & improvements can be found in the __[Final Report](/Sales Forecast and Churn Prediction_Final Report.docx)__ as well as conclusions and occured problems.

## Additional content
Addition content is included in the subfolders:
- Samples of SAP and the preprocessing in SQL can be tracked in the subfolders __[images/ERP-System+SQL](/ChurnProject/images/ERP-System%2BSQL)__ and __[SQL](/ChurnProject/SQL)__.
- In __[images](/ChurnProject/images)__ screenshots and plots from dataviz & modelling can be found.
- __[preprocessed](/ChurnProject/preprocessed)__ contains different variations of preprocessed data.
- In __[variables](/ChurnProject/variables)__ especially GridSearch results were saved as variables to save time when rerunning the kernel.


