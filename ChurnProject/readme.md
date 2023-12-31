# Churn Predictions of international health insurance contracts

## Introduction
The goal of this sub-project is to predict if (/probability that) a contract will be terminated by the customer. In particular:
1. Which contrcats have the highest probability to get terminated by the customer?
2. What are main global impacts on contract terminations?
3. What are individual factors for customers to terminate their contracts?

The classification problem can be described by the following ConfusionMatrix:
![ConfusionMatrix](https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/assets/134729968/f3122fbc-f17c-47c4-b3fa-b525955dd29c)

Therefore different kinds of contract related information were extracted from the ERP System SAP B1 and processed in SQL to create a pseudonimised csv file with features per contract. The most recent version is __[BDAE_DataMining_Policies_v2.csv](/ChurnProject/BDAE_DataMining_Policies_v2.csv)__.
![Data Collection Steps](https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/assets/134729968/bd8a03c6-9d32-47e8-af7d-ceb3ba21588a)

## Processing
The data got then further processed in 2 Jupyter Notebooks:
1. __[Churn_Introduction+DataExploration](/ChurnProject/Churn_Introduction%2BDataExploration.ipynb)__:
  - Introduction, first Overview, Data Exploration and first Modelling attempt to get main features with DecisionTree.
![Data Exploration Examples](https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/assets/134729968/5e656deb-0427-4c01-abfb-26eace06d0bf)


2. __[Churn_Preprocessing+Modelling](/ChurnProject/Churn_Preprocessing%2BModelling.ipynb)__:
  - Data Cleaning & Preprocessing, Data Visualisation & further exploration of correlations, cistributions & Co.
  - Train- & Test-Data creation, including alternative target variable for specific termination reasons only.
  - Classification predictions with XGBoost-, SupportVector- & RandomForest-Classifier on both target variables. Tuning attempts, comparison & interpretation with SHAP.
  -  Probability predictions with XGBoost, interpretation with SHAP.
  -  For all steps a lot of functions have been defined and ecluded into __[churn_helpers.py](/ChurnProject/churn_helpers.py)__
![Modelling Examples](https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/assets/134729968/3f2b57c6-ce0e-45d4-be61-477e9daac2f1)


## Results
In Chapter III of the __[Final Report](</Sales Forecast and Churn Prediction_Final Report.docx>)__ main preprocessing steps, data visualisations and modeelling results & improvements can be found as well as challenges and a conclusion.

## Additional content
Addition content is included in the subfolders:
- Samples of SAP and the preprocessing in SQL can be tracked in the subfolders __[images/ERP-System+SQL](/ChurnProject/images/ERP-System%2BSQL)__ and __[SQL](/ChurnProject/SQL)__.
- In __[images](/ChurnProject/images)__ screenshots and plots from dataviz & modelling can be found.
- __[preprocessed](/ChurnProject/preprocessed)__ contains different variations of preprocessed data.
- In __[variables](/ChurnProject/variables)__ especially GridSearch results were saved as variables to save time when rerunning the kernel.
- __[3_Churn_Prediction.py](/ChurnProject/3_Churn_Prediction.py)__ includes a first attempt of a streamlit demo of preprocessing and modelling.


