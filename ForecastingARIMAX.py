#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas openpyxl matplotlib statsmodels scikit-learn')


# In[4]:


import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import re


# In[2]:


warnings.filterwarnings("ignore")


# In[3]:


mrr = pd.read_excel('inputfile.xlsx', sheet_name='fct_org_consumption_monthly')


# In[4]:


org_df=pd.read_excel('inputfile.xlsx', sheet_name='dim_orgs')


# In[5]:


org_df['org_create_month']=pd.to_datetime(org_df['org_create_month'])


# In[6]:


org_df['org_create_month'] = org_df['org_create_month'].dt.tz_localize(None)


# In[ ]:


def reindex_and_fill(df):
    """
    Reindexes the DataFrame to fill in missing dates for each 'org_id' and forward fills the 'org_id' column.
    This ensures there are no gaps in the time series data for each organization.
    
    Args:
    df (pd.DataFrame): The input DataFrame containing 'org_id' and 'month' columns.
    
    Returns:
    pd.DataFrame: A DataFrame with missing dates reindexed and 'org_id' forward filled.
    """
    def reindex_group(group):
        start_date = group['month'].min()
        end_date = group['month'].max()
        all_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        group = group.set_index('month').reindex(all_dates).reset_index().rename(columns={'index': 'month'})
        group['org_id'] = group['org_id'].ffill()  # Fill the 'org_id' column
        return group
    
    df = df.groupby('org_id').apply(reindex_group).reset_index(drop=True)
    return df


# In[8]:


mrr = reindex_and_fill(mrr)


# In[9]:


mrr['total_mrr'] = mrr.groupby('org_id')['total_mrr'].transform(lambda group: group.ffill())


# In[10]:


merged_df = pd.merge(mrr, org_df, left_on=['org_id'], right_on=['org_id'])


# In[11]:


merged_df.shape


# In[12]:


merged_df['total_mrr']=merged_df['total_mrr'].round(2)


# In[ ]:


def calculate_month_diff(row):
    """
    This function calculates the difference in months between two dates in a row.
    It takes a row as input and computes the number of months between the 'month'
    column and the 'org_create_month' column. The function returns the total difference in months.
    
    Args:
    row (pd.Series): A row of data containing 'month' and 'org_create_month' columns.
    
    Returns:
    int: The difference in months between 'month' and 'org_create_month'.
    """
    return (row['month'].year - row['org_create_month'].year) * 12 + row['month'].month - row['org_create_month'].month


# In[14]:


merged_df['time_since_org_created'] = merged_df.apply(calculate_month_diff, axis=1)


# In[15]:


merged_df=merged_df.sort_values(['org_id','month'])
merged_df.set_index(['org_id', 'month'], inplace=True)


# In[ ]:


def split_data(df, id_col, y_col, exog_col, train_size=0.8):
    """
    Splits the input DataFrame into training and testing sets based on the specified identifier column.
    For each unique identifier, the function separates the data into dependent variable (y_col) and 
    exogenous variables (exog_col), and further splits these into training and testing sets based on the 
    specified train size.
    
    Args:
    df (pd.DataFrame): The input DataFrame containing the data to be split.
    id_col (str): The column name of the identifier used for splitting.
    y_col (str): The column name of the dependent variable.
    exog_col (list of str): A list of column names for the exogenous variables.
    train_size (float, optional): The proportion of the data to include in the training set (default is 0.8).

    Returns:
    dict: A dictionary where keys are unique identifiers, and values are tuples containing 
          training data (dependent and exogenous) and testing data (dependent and exogenous).
    """
    ids = df.index.get_level_values(id_col).unique()
    split_data = {}
    for id_ in ids:
        df_id = df.xs(id_, level=id_col)
        train_size_idx = int(len(df_id) * train_size)
        train_data = df_id.iloc[:train_size_idx][y_col]
        test_data = df_id.iloc[train_size_idx:][y_col]
        train_exog = df_id.iloc[:train_size_idx][exog_col]
        test_exog = df_id.iloc[train_size_idx:][exog_col]
        split_data[id_] = (train_data, test_data, train_exog, test_exog)
    return split_data


# In[ ]:


def find_best_arimax_model(data, exog, p_values, d_values, q_values, test_data, test_exog):
    """
    Finds the best ARIMAX model based on given parameters and evaluates them using AIC and MAPE.
    
    Args:
    data (pd.Series): Time series data for training the model.
    exog (pd.DataFrame): Exogenous variables used for training.
    p_values (list): List of p-values to test.
    d_values (list): List of d-values to test.
    q_values (list): List of q-values to test.
    test_data (pd.Series): Time series data for testing the model.
    test_exog (pd.DataFrame): Exogenous variables used for testing.

    Returns:
    list: A list of dictionaries containing the parameters and their corresponding AIC and MAPE scores.
    """
    results = []
    best_order = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(data, exog=exog, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    
                    # Forecast future values
                    forecast = model_fit.forecast(steps=len(test_data), exog=test_exog)
                    forecast = pd.Series(forecast, index=test_data.index)
                    
                    # Calculate MAPE
                    mape = mean_absolute_percentage_error(test_data, forecast)
                    
                    # Store results
                    results.append({
                        'p': p,
                        'd': d,
                        'q': q,
                        'aic': aic,
                        'mape': mape
                    })
                
                except:
                    continue

    return results


# In[ ]:


def forecast_for_all_ids(split_data_dict, p_values, d_values, q_values):
    """
    Forecasts future values for each unique ID in the provided data and returns the best results.

    Args:
    split_data_dict (dict): A dictionary where keys are unique IDs and values are tuples containing 
                            training data, testing data, training exogenous variables, and testing 
                            exogenous variables.
    p_values (list): List of p-values to test for the ARIMAX model.
    d_values (list): List of d-values to test for the ARIMAX model.
    q_values (list): List of q-values to test for the ARIMAX model.

    Returns:
    tuple: A dictionary containing forecasts for each ID and another dictionary containing the 
           best model parameters and performance metrics for each ID.
    """
    forecasts = {}
    best_results = {}

    for id_, (train_data, test_data, train_exog, test_exog) in split_data_dict.items():
        try:
            results = find_best_arimax_model(train_data, train_exog, p_values, d_values, q_values, test_data, test_exog)
            best_result = min(results, key=lambda x: x['mape'])
            print(f"Best Result for org_id {id_}: {best_result}")

            # Store the best result
            best_results[id_] = best_result

            # Run the model with the best parameters
            best_model = ARIMA(train_data, exog=train_exog, order=(best_result['p'], best_result['d'], best_result['q']))
            best_model_fit = best_model.fit()

            # Determine the number of periods to forecast until July 2024
            last_date = test_data.index[-1]
            target_date = pd.Timestamp('2024-07-01')
            periods_to_forecast = (target_date.year - last_date.year) * 12 + target_date.month - last_date.month

            # Prepare exogenous variable for the forecast period
            future_exog = np.arange(test_exog.iloc[-1] + 1, test_exog.iloc[-1] + periods_to_forecast + 1)
            exog = np.concatenate([test_exog, future_exog])

            # Calculate the forecasts
            forecast = best_model_fit.forecast(steps=len(exog), exog=exog)
            forecast = pd.Series(forecast, index=pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods_to_forecast, freq='MS'))
            forecast = forecast.round(2)

            # Last 3 months
            forecast = forecast.tail(3)

            forecasts[id_] = forecast
        except:
            continue

    return forecasts, best_results


# In[19]:


p_values = range(0, 3)
d_values = range(0, 3)
q_values = range(0, 3)


# In[20]:


split_data_dict = split_data(merged_df, 'org_id', 'total_mrr', 'time_since_org_created')


# In[21]:


forecasts, best_results = forecast_for_all_ids(split_data_dict, p_values, d_values, q_values)


# In[26]:


len(forecasts)==len(np.unique(org_df['org_id']))


# In[27]:


#### forecasts available for all org_ids


# In[23]:


# Initialize lists to store the data
ids = []
months = []
forecast_values = []

# Iterate through the forecasts dictionary
for id_, forecast in forecasts.items():
    for month, value in forecast.items():
        ids.append(id_)
        months.append(month)
        forecast_values.append(value)

# Create the DataFrame
forecast_df = pd.DataFrame({
    'org_id': ids,
    'month': months,
    'total_mrr': forecast_values
})


# In[31]:


forecast_df.shape


# In[33]:


np.unique(forecast_df['month'])


# In[35]:


forecast_df.to_csv('org_id_forecasts.csv',index=None)


# In[28]:


# Initialize lists to store the best results data
best_ids = []
best_ps = []
best_ds = []
best_qs = []
best_aics = []
best_mapes = []

# Iterate through the best_results dictionary
for id_, result in best_results.items():
    best_ids.append(id_)
    best_ps.append(result['p'])
    best_ds.append(result['d'])
    best_qs.append(result['q'])
    best_aics.append(result['aic'])
    best_mapes.append(result['mape'])

# Create the DataFrame
best_results_df = pd.DataFrame({
    'org_id': best_ids,
    'p': best_ps,
    'd': best_ds,
    'q': best_qs,
    'aic': best_aics,
    'mape': best_mapes
})


# In[29]:


best_results_df.shape


# In[30]:


best_results_df.head()


# In[ ]:




