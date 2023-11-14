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






