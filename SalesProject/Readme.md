# Sales_Classification

Sales Project - Prediction of Premium Amount with classification models

Project Github Repo: https://github.com/JonathanPablo/DataScientest_Sales-Churn_Project/tree/main/SalesProject
GitHub Repo: https://github.com/RumiyaAlMeri/Rumi

The goal of the subproject: 
- to find the best model for a class prediction for a future sum of a premium amount (grouped by month).

The classes were defined by the separation of a sum of a premium amount into 4 bins: XS, S, M and L. 

This subproject uses classification models such as KNN and SVM.

Both models gave a good result in terms of accuracy (was chosen as an evaluation score). 

KNN model with Minkowski metric performs the best. After the evaluation of the best k-value for the model, the k=3 and k=5 give the best results.

Still, from the business prespectve both models are hardly applicable as over time taking the company is successful, the premium amount will be growing and all future predictions will be found in the L-class.
Thus, the model should be adapted over time.
Moreover, the exact premium amount is forecasted in the company as it gives a more precise basis for a business plan. Only class will not help for a business plan calculation.

More details on the subproject in the final report. 

# Sales_TimeSerie

SALES Project - Time Series - Prediction of Premium Amount with different models

![Time_Series_1](https://github.com/tis294/Sales_TimeSeries/assets/125119694/1d71e7a1-ef6d-42db-b82d-0b25f4dc06b4)




General: 
The basic idea of the project was to predict the premium amount with different time series models. Several models were tested and evaluated.  On the one hand the classical SARIMA model with only the change of the premium Amount. An additive and multiplicative model was tested and the time series was made stationary. In addition, other models such as Multiple Linear Regression, Random Forest Tree and XGBoost were run. Only the temporal and lag features were used as features. The individual models were evaluated with R^2 and RSME. The possibility of the Forecast was inserted. A recursive approach was chosen. The next month was estimated and taken over into the model. Finally, time series splits and grid search CV were performed. Due to the different zones within the data, there is the possibility to adapt the data to the different zones. 

Basis: monthly premium amount data in the period 2014-2022. 
Choice between multiple zones.
Models: SARIMA, Multiple Linear Regression, Random Forest Tree, XGBoost
Features: Time Features (month, year, quarter), Lag Features (freely selectable)
Forecast: Recursive
Evaluation metrics: R^2, RSME
Optimization: Time Series Split

![Time_Series_2](https://github.com/tis294/Sales_TimeSeries/assets/125119694/05e68e9b-2711-4bdd-a77b-e6d694c076c9)




![3](https://github.com/tis294/Sales_TimeSeries/assets/125119694/51555914-c054-4708-83b8-946594e7398d)






