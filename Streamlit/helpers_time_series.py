

# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

import pandas as pd
import plotly.graph_objects as go

# Set Plotly to render in Jupyter Notebook
import plotly.io as pio
pio.renderers.default = 'notebook'

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

##### 1.1. choose_ZoneUnited

def choose_Zone(df,zoneUnited=None):
    """
    Filters a given DataFrame based on the 'Zone_united' column, according to the specified 'zoneUnited' value.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be filtered.
    zoneUnited (str or None, optional): The value to filter the 'Zone_united' column. Can be 'DE', 'CAN', or 'noCAN'.
                                        If None, no filtering will be applied.

    Returns:
    pandas.DataFrame: A filtered DataFrame containing rows where the 'Zone_united' column matches the specified value.
                      If 'zoneUnited' is None, the original DataFrame is returned without any filtering.
    """    
    if zoneUnited is not None:
        df = df[df['Zone_united']==zoneUnited]
    return df

##### 1.2. Group DF

def group_df_per_month(df, timerange=None, old=None, new=None):
    if old is not None or new is not None:
        if new is not None:
            df = df[df['Zone_united'] == new]
        if old is not None:
            df = df[df['Model_Zone'] == old]

    df_grouped = df.groupby('premiumMonth')['premiumAmount'].sum().to_frame()
    df_grouped.index = pd.to_datetime(df_grouped.index)

    if timerange is not None:
        df_grouped = df_grouped[df_grouped.index < timerange]

    return df_grouped


##### 1.3 Feature Selection

def choose_features(df_grouped, features=None, lag_count=12):
    """
    This function creates a DataFrame with selected features based on the provided data.

    Args:
        df_grouped (pd.DataFrame): A grouped DataFrame on which calculations will be performed.
        features (str, optional): The type of features to select ('time_features' or 'lag_features'). Default is None.
        lag_count (int, optional): The number of lag periods for lagging features. Default is 12.

    Returns:
        pd.DataFrame: A DataFrame with selected features according to the specified parameters.
    """

    # Copy the grouped DataFrame for manipulation
    df_features = df_grouped.copy()

    # Add columns for quarter, month, and year based on the index date
    df_features['quarter'] = df_features.index.quarter
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year

    # Create a mapping of target values for lagging features
    target_map = df_features['premiumAmount'].to_dict()

    # Generate lagging features for each lag period
    for lag in range(1, lag_count + 1):
        df_features['lag' + str(lag)] = (df_features.index - pd.DateOffset(months=lag)).map(target_map)

    # Add a 'Time' feature representing the sequence of records
    df_features['Time'] = np.arange(len(df_features.index))

    # Select features based on the 'features' parameter
    if features == 'time_features':
        return df_features[['premiumAmount', 'quarter', 'month', 'year', 'Time']]
    if features == 'lag_features':
        return df_features.drop(['quarter', 'month', 'year'], axis=1)

    # Return the DataFrame if no specific features are selected
    return df_features


#### 1.4 Check Stationary

import statsmodels.api as sm

def check_stationary(data):
    _, p_value, _, _, _, _  = sm.tsa.stattools.adfuller(data)
    return round(p_value,2)  # p-vvalue below 1% the time series can be considered stationnary.

##### 1.5. Split in Train and Test

def train_test(df_features, cut_date):
    """
    This function splits a DataFrame into training and testing sets based on the specified cut-off date.

    Args:
        df_features (pd.DataFrame): The DataFrame containing the features and target values.
        cut_date (str or pd.Timestamp): The date that separates the training and testing periods.

    Returns:
        pd.DataFrame, pd.DataFrame: Two DataFrames representing the training and testing sets.
    """
    # Create the training set containing rows before the cut-off date
    train = df_features.loc[df_features.index < cut_date]

    # Create the testing set containing rows on or after the cut-off date
    test = df_features.loc[df_features.index >= cut_date]

    return train, test

#### 1.6 Stationary

def stationary_log(df_feature,kind=None):
    if kind == 'log':
        df_feature['premiumAmount']= np.log(df_feature['premiumAmount'])
    if kind == 'diff':
        df_feature['premiumAmount'] = df_feature['premiumAmount'].diff()
        df_feature.dropna(inplace=True)
    return df_feature

##### 2.1 Lineare Regression



def linear_regression(train, test, var):
    """
    This function performs linear regression modeling using the provided training data and target variable.

    Args:
        train (pd.DataFrame): The training data containing features and target variable.
        test (pd.DataFrame): The testing data containing features and target variable.
        var (str): The name of the target variable to be predicted.

    Returns:
        LinearRegression, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series:
        The trained linear regression model, training feature matrix, training target vector,
        testing feature matrix, and testing target vector.
    """
    # Extract features and target variable for training
    X_train = train.loc[:, ['Time']]
    y_train = train.loc[:, var]

    # Extract features and target variable for testing
    X_test = test.loc[:, ['Time']]
    y_test = test.loc[:, var]

    # Initialize and train the linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Return the trained model and the data splits
    return lr_model, X_train, y_train, X_test, y_test

#### 2.2 Multiple Lineare Regression






#### 2.3 Forecast Regression


def forecast_regression(df_features, period=6, lag_count=12, model=None):
    """
    This function performs regression-based forecasting using linear regression for a specified number of periods.

    Args:
        df_features (pd.DataFrame): The DataFrame containing features and historical target values.
        period (int, optional): The number of future periods to forecast. Default is 6.

    Returns:
        pd.DataFrame: A DataFrame with historical and forecasted premium amounts.
    """
    df_forecast = df_features.copy()

    # X_train = df_forecast.drop(['premiumAmount'], axis=1)  # features
    # y_train = df_forecast.loc[:, 'premiumAmount']  # target
    #
    # if model == 'mlr':
    #    ml = LinearRegression()
    #    ml.fit(X_train, y_train)
    # if model == 'rft':
    #    ml = RFE(RandomForestRegressor(n_estimators=1000, random_state=1), n_features_to_select=15)
    #    ml = ml.fit(X_train, y_train)
    # if model == 'xgboost':
    #    ml = xgb.XGBRegressor(base_score=0.5, booster = 'gbtree',
    #                      n_estimators=1000,
    #                      early_stopping_rounds=50,
    #                      objective='reg:squarederror',
    #                      max_depth=10,
    #                      learning_rate=0.01)
    #    ml = ml.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)])
    #
    ml = model
    pred_values = []

    for i in range(period):
        index_date = df_forecast.index.max() + pd.DateOffset(months=1)
        future_df = pd.DataFrame(index=[index_date])
        future_df['isFuture'] = True
        df_forecast['isFuture'] = False
        df_and_future = pd.concat([df_forecast, future_df])
        df_and_future = choose_features(df_and_future, lag_count=lag_count)
        future_w_features = df_and_future.query('isFuture').copy()
        pred = ml.predict(future_w_features.drop(['premiumAmount', 'isFuture'], axis=1))
        future_w_features['premiumAmount'] = pred
        pred_values.append(pred)
        df_forecast = pd.concat([df_forecast, future_w_features])

    return df_forecast.tail(period)


#### 2.4. Evaluate Regression


from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# def train_test(df_features, n_splits=5):
#    """
#    This function splits a DataFrame into training and testing sets based on the specified cut-off date.
#    It uses time series splitting to create multiple training and testing sets.
#
#    Args:
#        df_features (pd.DataFrame): The DataFrame containing the features and target values.
#        cut_date (str or pd.Timestamp): The date that separates the training and testing periods.
#        n_splits (int, optional): Number of splits for time series cross-validation. Default is 5.
#
#    Returns:
#        List[Tuple[pd.DataFrame, pd.DataFrame]]: A list of tuples, each containing training and testing sets.
#    """
#    # Ensure the DataFrame is sorted by the index (assuming the index is a time-based column)
#    df_features = df_features.sort_index()
#
#    # Initialize TimeSeriesSplit
#    tscv = TimeSeriesSplit(n_splits=n_splits)
#
#    # Create an empty list to store train-test splits
#    splits = []
#
#    # Perform time series splitting
#    for train_index, test_index in tscv.split(df_features):
#        train, test = df_features.iloc[train_index], df_features.iloc[test_index]
#        splits.append((train, test))
#
#    return splits, tscv
#

def multiple_linear_regression_2(train, test, var):
    """
    This function performs multiple linear regression modeling using time series cross-validation
    and grid search cross-validation for hyperparameter tuning.

    Args:
        train (pd.DataFrame): The training data containing features and target variable.
        test (pd.DataFrame): The testing data containing features and target variable.
        var (str): The name of the target variable to be predicted.
        tscv: TimeSeriesSplit object for cross-validation.

    Returns:
        LinearRegression, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series:
        The trained multiple linear regression model, training feature matrix, training target vector,
        testing feature matrix, and testing target vector.
    """

    # Extract features and target variables
    X_train, y_train = train.drop(var, axis=1), train[var]
    X_test, y_test = test.drop(var, axis=1), test[var]

    # Initialize linear regression model
    lr_model = LinearRegression()

    # Parameter grid for hyperparameter tuning
    param_grid = {
        'fit_intercept': [True, False],  # Whether to calculate the intercept for this model
        'copy_X': [True, False],  # If True, X will be copied; else, it may be overwritten
    }

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_lr_model = grid_search.best_estimator_

    # Train the best model on the entire training set
    best_lr_model.fit(X_train, y_train)

    # Return the trained model and the data splits
    return best_lr_model, X_train, y_train, X_test, y_test, grid_search.best_params_


def evaluate_regression_model(model, X_train, y_train, X_test, y_test, best_params):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    # display(X_train)
    # Create a DataFrame to store the results
    results = pd.DataFrame({

        'Metric': ['R²', 'RMSE'],
        'Train': [r2_train, rmse_train],
        'Test': [r2_test, rmse_test],
    })
    # display(best_params)
    # Add a new column for each key-value pair in best_params
    for key, value in best_params.items():
        results[key] = value

    return results


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV


def random_forest_tree(train, test, var):
    """
    This function performs random forest tree modeling using the provided training data and target variable.

    Args:
        train (pd.DataFrame): The training data containing features and target variable.
        test (pd.DataFrame): The testing data containing features and target variable.
        var (str): The name of the target variable to be predicted.

    Returns:
        RFE, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series:
        The trained Random Forest model, training feature matrix, training target vector,
        testing feature matrix, and testing target vector.
    """
    # Extract features and target variable for training
    X_train = train.drop([var], axis=1)
    y_train = train.loc[:, var]

    # Extract features and target variable for testing
    X_test = test.drop([var], axis=1)
    y_test = test.loc[:, var]

    # Parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [None, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    # Initialize Random Forest regressor
    rf_clf = RandomForestRegressor(random_state=1)

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid,  scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best Random Forest model from the grid search
    best_rf_model = grid_search.best_estimator_

    # Train the best model on the entire training set
    best_rf_model.fit(X_train, y_train)

    # Perform feature selection using RFE with the best model
    rfe = RFE(best_rf_model)
    rfe = rfe.fit(X_train, y_train)

    # Return the trained model and the data splits
    return rfe, X_train, y_train, X_test, y_test, grid_search.best_params_


import xgboost as xgb
from sklearn.model_selection import GridSearchCV


def xgboost(train, test, var):
    """
    This function performs xgboooz modeling using the provided training data and target variable.

    Args:
        train (pd.DataFrame): The training data containing features and target variable.
        test (pd.DataFrame): The testing data containing features and target variable.
        var (str): The name of the target variable to be predicted.

    Returns:
        XGBoost, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series:
        The trained XGBoost model, training feature matrix, training target vector,
        testing feature matrix, and testing target vector.
    """
    # Extract features and target variable for training
    X_train = train.drop([var], axis=1)
    y_train = train.loc[:, var]

    # Extract features and target variable for testing
    X_test = test.drop([var], axis=1)
    y_test = test.loc[:, var]

    # Parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100],
        'max_depth': [5],
        'learning_rate': [0.05],
        'subsample': [0.8],
    }

    # Initialize XGBoost regressor
    xgb_clf = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', base_score=0.5)

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid,  scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best XGBoost model from the grid search
    best_xgb_model = grid_search.best_estimator_

    # Train the best model on the entire training set
    best_xgb_model.fit(X_train, y_train)

    # Return the trained model and the data splits
    return best_xgb_model, X_train, y_train, X_test, y_test, grid_search.best_params_


import plotly.graph_objects as go

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

import plotly.graph_objects as go
import streamlit as st
import pandas as pd


def graph(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, future_pred=None, title=None):
    # Define colors
    colors = {
        'background': '#D3D3D3',
        'train': '#174A7E',
        'train_pred': '#646369',
        'test_pred': '#9ABB59 ',
        'test': '#0C8040',
        'future': 'black'
    }

    # Create a Plotly figure
    fig = go.Figure()

    # Plot training data
    fig.add_trace(go.Scatter(x=X_train.index, y=y_train, mode='lines', line=dict(color=colors['train']), name='Train'))

    # Plot training predictions
    fig.add_trace(
        go.Scatter(x=X_train.index, y=y_train_pred, mode='lines', line=dict(color=colors['train_pred'], dash='dash'),
                   name='Train Prediction'))

    # Plot test predictions
    fig.add_trace(
        go.Scatter(x=X_test.index, y=y_test_pred, mode='lines', line=dict(color=colors['test_pred'], dash='dash'),
                   name='Test Prediction'))

    # Plot actual test data
    fig.add_trace(go.Scatter(x=X_test.index, y=y_test, mode='lines', line=dict(color=colors['test']), name='Test'))

    # Plot future predictions if provided
    if future_pred is not None:
        fig.add_trace(go.Scatter(x=future_pred.index, y=future_pred['premiumAmount'], mode='lines',
                                 line=dict(color=colors['future']), name='Future'))

    # Customize layout
    fig.update_layout(
        legend=dict(x=0.05, y=1.15, font=dict(color='black')),
        title=title,
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(family='Arial', size=14, color='black'),
        xaxis=dict(title='Time', gridcolor='black', tickfont=dict(color='black')),
        yaxis=dict(title='Premium Amount', gridcolor='black', tickfont=dict(color='black'))
    )

    # Show the plot
    return fig


# Beispielaufruf:
# graph(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, future_pred, title='Ihr Titel')


# Beispielaufruf:
# graph(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, future_pred, title='Ihr Titel')


# Usage
def plot_timeseries(df):
    # Define colors
    colors = {
        'background': '#D3D3D3',
        'train': '#174A7E',
        'train_pred': '#646369',
        'test_pred': '#9ABB59 ',
        'test': '#0C8040',
        'color': 'black'
    }
        # Create a Plotly figure
    fig = go.Figure()



    # Plot training data
    fig.add_trace(go.Scatter(x=df.index, y=df['premiumAmount'], mode='lines', line=dict(color=colors['color'])))

    fig.update_layout(
        legend=dict(x=0.05, y=1.15, font=dict(color='black')),
        title = 'Premium Amount',
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(family='Arial', size=14, color='black'),
        xaxis=dict(title='Time', gridcolor='black', tickfont=dict(color='black')),
        yaxis=dict(title='Premium Amount', gridcolor='black', tickfont=dict(color='black'))
    )




    return fig

def plot_acf_pacf(data):
    fig, ax = plt.subplots(2,1, figsize =(18, 10))
    fig = sm.graphics.tsa.plot_acf(data, lags=24, ax=ax[0])
    fig = sm.graphics.tsa.plot_pacf(data, lags=24, ax=ax[1])
    return fig

def stationary_log(df_feature,kind=None):
    if kind == 'log':
        df_feature['premiumAmount']= np.log(df_feature['premiumAmount'])
    if kind == 'diff':
        df_feature['premiumAmount'] = df_feature['premiumAmount'].diff()
        df_feature.dropna(inplace=True)
    return df_feature

