# -*- coding: utf-8 -*-
"""
Created on 2023-10-15

@author: JL
"""

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Conclusion",layout="wide")

# define subfolder for images
if 'image_folder' not in st.session_state:
  st.session_state.image_folder = 'images/streamlit/'

'''
## Results

### Sales Forecast:
- Time series: 
    - Best results over all Zones with Random Forest Regressor and XGboost. XGBoost is faster.
    - Solid results on train & test data
    - Forecast is generally should be treated with caution 
- Classification: 
    - KNN-model with Manhattan metric shows the best results
    - still the model cannot be applied in the business context.
'''

'''### Churn Prediction:'''

col1, col2 = st.columns((2,1))

churn_text= '''
            - Solid results, XGBClassifier most suitable for interpretation
            - Interesting Insights from SHAP
            - Too much focus on effective end
            - Limited business relevant success
            '''

col1.markdown(churn_text)
# Open image of churn 
image_path = st.session_state.image_folder + 'Churn_SHAP.png'
try:
    image = Image.open(image_path)
    col2.image(image, caption='Example of Churn Interpretation')
except FileNotFoundError:
    col2.error(f"Image file '{image_path}' not found. Please check the file path.")

'''
## Challenges
- project management
- real world data
- preprocessing & interpretation
- human made decisions
'''
