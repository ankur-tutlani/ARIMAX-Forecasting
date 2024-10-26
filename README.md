# ARIMAX Model Forecasting

This repository contains code for forecasting Monthly Recurring Revenue (MRR) using ARIMAX models. The code reads data from an Excel file, preprocesses it, and applies ARIMAX models to forecast future values. It is applicable to forecast any continous scale feature.

## Requirements

Install the required Python packages:
```bash
pip install pandas openpyxl matplotlib statsmodels scikit-learn
```

## Usage
1. Data Preparation:
- Load data from inputfile.xlsx.
- Reindex and fill missing dates for each organization.
- Merge dataframes and calculate the time since each organization was created.

2. Model Training and Forecasting:
- Split data into training and testing sets.
- Find the best ARIMAX model for each organization based on AIC and MAPE.
- Forecast future MRR values until July 2024.

3. Results:
- Store forecasts and best model parameters in CSV files.

## Functions

- reindex_and_fill(df): Reindexes and fills missing dates.
- calculate_month_diff(row): Calculates the difference in months between two dates.
- split_data(df, id_col, y_col, exog_col, train_size): Splits data into training and testing sets.
- find_best_arimax_model(data, exog, p_values, d_values, q_values, test_data, test_exog): Finds the best ARIMAX model.
- forecast_for_all_ids(split_data_dict, p_values, d_values, q_values): Forecasts future values for each organization.

## Output

- org_id_forecasts.csv: Contains forecasted MRR values.
- best_results_df.csv: Contains the best model parameters and performance metrics.

## License
This project is licensed under the MIT License.
