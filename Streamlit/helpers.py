#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   

#1.1
def convert_columns_to_datetime(df, columns):
    """
    Converts specified columns of dtype 'object' or 'period' in a Pandas DataFrame to dtype 'datetime'.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columns (list): A list of column names to convert.

    Returns:
        pandas.DataFrame: The modified DataFrame with converted columns.
    """
    for column in columns:
        if df[column].dtype == "object":
            try:
                df[column] = pd.to_datetime(df[column])
            except ValueError:
                print(f"Error converting column '{column}' to datetime. Invalid date format.")
        elif df[column].dtype.name.startswith("period"):
            try:
                df[column] = pd.to_datetime(df[column].astype(str))
            except ValueError:
                print(f"Error converting column '{column}' to datetime. Invalid date format.")
    return df

#1.1.1
def auto_convert_columns_to_datetime(df):
    """
    Converts columns of dtype 'object' or 'period' in a Pandas DataFrame to dtype 'datetime' if possible.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.

    Returns:
        pandas.DataFrame: The modified DataFrame with converted columns.
    """
    for column in df.columns:
        if df[column].dtype == "object":
            try:
                df[column] = pd.to_datetime(df[column])
            except ValueError:
                print(f"Error converting column '{column}' to datetime. Invalid date format.")
        elif df[column].dtype.name.startswith("period"):
            try:
                df[column] = pd.to_datetime(df[column].astype(str))
            except ValueError:
                print(f"Error converting column '{column}' to datetime. Invalid date format.")
    
    return df

#1.2
def NaN_col_info(df, nan_col, dist_col):
    '''
    Create Graphs to compare infos of rows with NaN-value in input column with Not-NaN values.

    Input: 
    - nan_col = column with NaN values 
    - dist_col = column to plot the distribution for

    Return: 
    Graphs of dist_col distribution, separated by NaN & Not-NaN values of nan_col
    '''
    # Create main figure
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))
    fig.suptitle(f'{dist_col} Distribution for NaN and Not-NaN values of {nan_col} column')

    # Plotting distribution of policy_startDate for NaN & not-NaN
    sns.histplot(df[df[nan_col].isna() == False][dist_col], kde=True, bins=10, color='blue', label='Not-NaN', ax=ax1) #Not-NaN -> blue
    sns.histplot(df[df[nan_col].isna() == True][dist_col], kde=True, bins=10, color='red', label='NaN', ax=ax2) #NaN -> red
     
    ax1.set_ylabel('Distribution of Not-NaN rows')
    ax2.set_ylabel('Distribution of NaN rows')
    ax2.set_xlabel(dist_col)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()

#1.3
def plot_distributions(df, nan_col, comp_cols):
    """
    Generates a graph showing the distributions of specified columns in 'comp_cols'
    based on NaN values and non-NaN values of the 'nan_col' column.

    Args:
        nan_col (str): The name of the column with NaN values.
        comp_cols (list): A list of column names to analyze.
        df (pandas.DataFrame): The DataFrame containing the data. (Default: df)
        
    Returns:
        None (displays a graph)

    """
    nan_values = df[df[nan_col].isna()]
    non_nan_values = df[df[nan_col].notna()]
    
    for col in comp_cols:
        nan_dist = nan_values[col].value_counts(normalize=True)
        non_nan_dist = non_nan_values[col].value_counts(normalize=True)
        
        plt.figure(figsize=(10, 6))
        plt.bar(nan_dist.index, nan_dist.values, label='NaN Values', alpha=0.5)
        plt.bar(non_nan_dist.index, non_nan_dist.values, label='Non-NaN Values', alpha=0.5)
        plt.xlabel(col)
        plt.ylabel('Distribution')
        plt.title(f'Distribution of {col} for NaN and Non-NaN Values of {nan_col}')
        plt.legend()
        plt.show()
        
#1.4
def get_unique_values(df, columns=['Nation', 'premium_Country']):
    """
    Get unique values from specified columns in a DataFrame.

    Parameters:
        df (pandas.DataFrame, optional): Input DataFrame. (Default: df)
        columns (list, optional): List of column names. Defaults to ['Nation', 'premium_Country'].

    Returns:
        list: List of unique values from the specified columns.

    Example Usage:
        unique_values = get_unique_values(columns=['Nation', 'premium_Country'])
        print("Unique Values:", unique_values)
    """
    unique_values = df[columns].values.ravel()
    unique_values = pd.unique(unique_values).tolist()
    return unique_values
        
#1.5 
import requests

def generate_country_region_mapping(iso_codes):
    """
    Generates a dictionary mapping country ISO codes to their corresponding regions.

    Parameters:
        iso_codes (list): List of unique country ISO codes.

    Returns:
        dict: Dictionary mapping ISO codes to regions.

    Example Usage:
        iso_codes = ['PL', 'DE', 'RO', ...]  # List of ISO codes
        mapping = generate_country_region_mapping(iso_codes)
        print("Country to Region Mapping:", mapping)
    """
    mapping = {}

    # Special feature: 'DE' has its own region called 'DE'
    mapping['DE'] = 'DE'

    # Special feature: 'XX' has its own region called 'XX'
    mapping['XX'] = 'XX'

    # Make a request to the REST Countries API
    response = requests.get('https://restcountries.com/v2/all')

    if response.status_code == 200:
        countries_data = response.json()

        # Iterate over the countries data and extract ISO code and region information
        for country_data in countries_data:
            iso_code = country_data['alpha2Code']
            region = country_data['region']

            if iso_code in iso_codes:
                if iso_code == 'DE':
                    mapping[iso_code] = 'DE'
                elif iso_code == 'XX':
                    mapping[iso_code] = 'XX'
                else:
                    mapping[iso_code] = region

    return mapping

#1.6
def premium_status_timeline(df, index):
    '''
    Input: 
    df
    index = number of row in df sorted by sum of premiumAmount with status_code = 'S'
    
    Output:
    prints: df with sum of premium amounts of ContractID for each status
    returns: df with all premium amounts of ContractID, their status and premiumMonth, sorted by premiumMonth
    '''
    #check if input index is in range
    n = len(df.loc[df.status_code == 'S'][['ContractID']])
    if index not in range(-n, n):
        return 'index out of range'
    else:
        #create df with contractIDs ordered by sum of premiumAmount of cancelled lines 
        S_sums = df.loc[df.status_code == 'S'][['ContractID', 'premiumAmount']].groupby(['ContractID']).sum('premiumAmount').sort_values('premiumAmount', ascending=False)

        #get contractID of input index
        ID = S_sums.index[index]

        #print premium overview per status for this ID
        print(f'premiumAmount overview for contract ID {ID}:\n',
            df.loc[df.ContractID == ID][['premiumMonth', 'status_code', 'premiumAmount']].groupby(['status_code']).sum('premiumAmount'))

        #create and return overview of premiums for this ID
        premiums = df.loc[df.ContractID == ID][['premiumMonth', 'status_code', 'premiumAmount']].sort_values('premiumMonth')
        premiums.set_index('premiumMonth')
        return premiums
    
#1.7
def groupby_and_apply(df, group_cols, apply):
    """
    Group a pandas DataFrame by specified columns and apply functions to specified columns.
    
    Input:
    - df: pandas DataFrame.
    - group_cols: list of column names to group by.
    - apply: dictionary with column names as keys and corresponding functions as values.
    
    Return:
    - result_df: pandas DataFrame with grouped data and applied functions.
    """
    result_df = df.groupby(group_cols).agg(apply).reset_index()
    
    # Rename the columns with key_value format
    result_df.columns = [f"{apply[col]}_{col}" if col in apply else col for col in result_df.columns]
    
    return result_df

#1.8
def plot_grouped_data(df, group_cols, apply, y=None, kind='line', x='premiumMonth', hue=None):
    """
    Group a pandas DataFrame by specified columns, apply functions to specified columns, and plot the result.
    
    Input:
    - df: pandas DataFrame.
    - group_cols: list of column names to group by.
    - apply: dictionary with column names as keys and corresponding functions as values.
    - y: list of column names to use for the y-axis (default: None, uses all columns except group_cols).
    - kind: type of plot to generate (default: 'line'). Other options: 'bar', 'scatter', etc.
    - x: column name to use as the x-axis (default: 'premiumMonth').
    - hue: column name to use for distinguishing plot colors (default: None).
    
    Example Usage:
    df = pd.DataFrame({'premiumMonth': [1, 1, 2, 2],
                       'premiumAmount': [5, 6, 7, 8],
                       'ContractID': [9, 10, 11, 12]})
    group_cols = ['premiumMonth']
    apply = {'premiumAmount': 'sum', 'ContractID': 'count'}
    plot_grouped_data(df, group_cols, apply, y=['premiumAmount_sum', 'ContractID_count'], kind='line', x='premiumMonth')
    
    Return:
    - None (displays the plot).
    """
    result_df = groupby_and_apply(df, group_cols, apply)
    
    if y is None:
        #y = result_df.columns[1:]
        y = result_df.columns[len(group_cols):]
    
    num_keys = len(result_df.columns) - len(group_cols)
    if num_keys == 1:
        fig, ax = plt.subplots(figsize=(20, 10))
        result_df.plot(x=x, y=y, kind=kind, ax=ax)
        ax.set_title(f"Timeline of {y[0]} in reference to {x}", fontsize=18)
    elif num_keys == 2:
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax2 = ax1.twinx()
        
        if hue:
            result_df.plot(x=x, y=y[0], kind=kind, ax=ax1, legend=False, color='blue')
            result_df.plot(x=x, y=y[1], kind=kind, ax=ax2, legend=False, color='red')
        else:
            result_df.plot(x=x, y=y[0], kind=kind, ax=ax1, legend=False, color='blue')
            result_df.plot(x=x, y=y[1], kind=kind, ax=ax2, legend=False, color='red')
            
        ax1.set_ylabel(y[0], fontsize=14)
        ax2.set_ylabel(y[1], fontsize=14)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.set_title(f"Timeline of {y[0]} and {y[1]} in reference to {x}", fontsize=18)
    else:
        raise ValueError("Number of keys in 'apply' must be 1 or 2.")
    
    plt.xlabel(x, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    
    plt.show()
    
#1.8 similar to 1.7, but with possibility to distinguish between unique values of another column
def plot_line_plots(df, x='premiumMonth', hue=None, y=['premiumAmount_sum', 'ContractID_count'], kind='line'):
    """
    Plots line plots on a single graph based on the provided DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        x (str): The column name to be used as the x-axis of the plot. Default: 'premiumMonth'.
        hue (str): The column name to differentiate the lines by different values. Default: None.
        y (list): The list of column names to be used as the y-axis of the plot.
                  If len(y) = 1, it creates a single y-axis. If len(y) = 2, it creates a secondary y-axis.
                  Default: ['premiumAmount_sum', 'ContractID_count'].
        kind (str): The type of plot to be created. Options: 'line' (default), 'scatter', 'bar', 'area', etc.
                    This parameter is passed to the plot() function of pandas DataFrame.
    
    Returns:
        None
    
    Example:
        plot_line_plots(df, x='premiumMonth', hue='ZoneDesc', y=['premiumAmount_sum', 'ContractID_count'], kind='line')
    """
    
    if len(y) == 1:
        fig, ax = plt.subplots()
        if hue is not None:
            for hue_val in df[hue].unique():
                df_hue = df[df[hue] == hue_val]
                ax.plot(df_hue[x], df_hue[y[0]], label=hue_val)
        else:
            ax.plot(df[x], df[y[0]])
        ax.set_ylabel(y[0])
    elif len(y) == 2:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        if hue is not None:
            for hue_val in df[hue].unique():
                df_hue = df[df[hue] == hue_val]
                ax1.plot(df_hue[x], df_hue[y[0]], label=hue_val)
                ax2.plot(df_hue[x], df_hue[y[1]], label=hue_val)
        else:
            ax1.plot(df[x], df[y[0]])
            ax2.plot(df[x], df[y[1]])
        ax1.set_ylabel(y[0])
        ax2.set_ylabel(y[1])
    else:
        raise ValueError("Invalid number of y-axis values. Please provide 1 or 2 y-axis values.")

    plt.xlabel(x)
    plt.title(f"{kind.capitalize()} Plot - {x} vs {', '.join(y)}")

    plt.legend()
    plt.show()
    
#1.9 Overview over distribution and influence of column values
def plot_subplots(df, input_column, other_columns=None, groupby_column=None, plot_type='line', functions=None):
    """
    Function to plot subplots for a given input column in a pandas DataFrame.

    Parameters:
        - df (pandas.DataFrame): The DataFrame containing the data.
        - input_column (str): The column for which to plot the subplots.
        - other_columns (list, optional): A list of other columns on which to apply functions.
        - groupby_column (str, optional): The column to group by. Default is input_column.
        - plot_type (str, optional): The type of plot to use for the second subplot. Can be either 'line' or 'bar'.
        - functions (dict, optional): A dictionary of column names and corresponding functions to apply.

    Returns:
        None
    """

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Plot distribution of input column
    axes[0].hist(df[input_column])
    axes[0].set_title(f"Distribution of {input_column}")

    # Set groupby column
    if groupby_column is None:
        groupby_column = input_column

    # Validate other_columns and functions lengths
    if other_columns and functions:
        if len(other_columns) != len(functions):
            raise ValueError("Number of columns in 'other_columns' should be equal to the number of functions in 'functions'.")

    # Apply functions on other columns and group by input column
    if other_columns and functions:
        # Group data by input column and apply functions on other columns
        grouped_data = df.groupby(groupby_column)[other_columns].agg(functions)

        # Plot the grouped data
        if len(other_columns) == 2:
            ax1 = grouped_data[other_columns[0]].plot(ax=axes[1], marker='o')
            ax2 = ax1.twinx()
            grouped_data[other_columns[1]].plot(ax=ax2, marker='s', color='r')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

            # Label the y-axis with column and function
            column1, column2 = other_columns
            function1, function2 = functions[column1], functions[column2]
            ax1.set_ylabel(f"{column1}_{function1}")
            ax2.set_ylabel(f"{column2}_{function2}")
        else:
            if plot_type == 'line':
                grouped_data.plot.line(ax=axes[1])
            elif plot_type == 'bar':
                grouped_data.plot.bar(ax=axes[1])
            else:
                print("Invalid plot type. Please choose either 'line' or 'bar'.")
                return

            # Label the y-axis with column and function
            for column in other_columns:
                function = functions[column]
                axes[1].set_ylabel(f"{column}_{function}")

        axes[1].set_xlabel(input_column)

    # Adjust spacing and display the plot
    plt.tight_layout()
    plt.show()
