# -*- coding: utf-8 -*-
"""
Created on 2023-10-15

@author: JL
"""

import streamlit as st

st.set_page_config(page_title="Conclusion",layout="wide")
'''
## Results

### Sales Forecast:
- Time series: Old Model Zone ML like XGBoost or Random Forest Regressor are suitable. For New Model Zone only Mulitple Linear Regression is suitable. 
- Classification: KNN-model with Manhattan metric shows the best results; still the model cannot be applied in the business context.

### Churn Prediction:
- Solid results, XGBClassifier most suitable for interpretation
- Too much focus on effective end
- Limited business relevant success

## Challenges
- project management
- real world data
- preprocessing & interpretation
- human made decisions
'''
