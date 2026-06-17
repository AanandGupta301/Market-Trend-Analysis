"""
Feature Engineering Module

This module contains functions for creating features from time series data
for agricultural price prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_lag_features(df, column='price_rupees_per_kg', lags=[1, 7, 30]):
    """
    Create lag features for time series prediction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to create lags for
    lags : list
        List of lag periods
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features added
    """
    df = df.copy()
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df


def create_rolling_features(df, column='price_rupees_per_kg', windows=[7, 30]):
    """
    Create rolling window statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to create rolling features for
    windows : list
        List of window sizes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling features added
    """
    df = df.copy()
    for window in windows:
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
        df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
    return df


def create_cyclical_features(df):
    """
    Create sin/cos transformations for cyclical time features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with temporal columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cyclical features added
    """
    df = df.copy()
    
    # Month cyclical encoding (12 months)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Day of week cyclical encoding (7 days)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Day of year cyclical encoding (365 days)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    return df


def engineer_all_features(df):
    """
    Apply all feature engineering steps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Create lag features
    df = create_lag_features(df, lags=[1, 7, 30])
    
    # Create rolling features
    df = create_rolling_features(df, windows=[7, 30])
    
    # Create cyclical features
    df = create_cyclical_features(df)
    
    # Drop rows with NaN values (due to lag and rolling operations)
    df = df.dropna()
    
    return df


def prepare_train_test_split(df, test_year=2022):
    """
    Split data into train and test sets based on time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'year' column
    test_year : int
        Year to use for test set
        
    Returns:
    --------
    tuple
        (train_df, test_df)
    """
    train_df = df[df['year'] < test_year].copy()
    test_df = df[df['year'] == test_year].copy()
    
    return train_df, test_df


def create_sequences(data, target, lookback=30):
    """
    Create sequences for LSTM training.
    
    Parameters:
    -----------
    data : np.array
        Feature data
    target : np.array
        Target data
    lookback : int
        Number of time steps to look back
        
    Returns:
    --------
    tuple
        (X, y) where X has shape (samples, lookback, features)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("Feature Engineering Module")
    print("This module provides functions for:")
    print("- Creating lag features")
    print("- Creating rolling window statistics")
    print("- Creating cyclical time features")
    print("- Preparing train/test splits")
    print("- Creating sequences for LSTM")
