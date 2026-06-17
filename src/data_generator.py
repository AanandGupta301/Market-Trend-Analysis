"""
Synthetic Onion Price Dataset Generator

This module generates a realistic synthetic dataset for onion prices in India
from 2018 to 2022 with daily granularity. The dataset includes multiple
realistic components that simulate real market behavior.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_onion_price_data(start_date='2022-01-01', end_date='2025-12-31', seed=42):
    """
    Generate synthetic onion price dataset with realistic market components.
    
    Components:
    1. Long-term upward trend: Simulates inflation and market growth
    2. Annual seasonality: Reflects harvest cycles (lower prices during harvest)
    3. Monthly seasonality: Captures within-year variations
    4. Random shocks: Simulates weather events, policy changes, supply disruptions
    5. Autoregressive noise: Adds realistic price persistence
    6. Promotional effects: Price drops during promotional periods
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format (default: 2022-01-01)
    end_date : str
        End date in 'YYYY-MM-DD' format (default: 2025-12-31)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, price_rupees_per_kg, volume_quintals, 
        promo_flag, year, month, day_of_week, day_of_year, week_of_year
    """
    
    np.random.seed(seed)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Base price (starting point - 2022 level)
    base_price = 35.0  # Rupees per kg (higher baseline for 2022)
    
    # 1. Long-term upward trend (inflation ~6% per year for recent period)
    trend = np.linspace(0, 8, n_days)  # Gradual increase over 4 years
    
    # 2. Annual seasonality (harvest cycles)
    # Onions are typically harvested in Oct-Nov (lower prices) and Feb-Mar
    annual_cycle = 5 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 - np.pi/2)
    
    # 3. Monthly seasonality (within-year variations)
    monthly_cycle = 2 * np.sin(2 * np.pi * np.arange(n_days) / 30.44)
    
    # 4. Random shocks (weather events, policy changes)
    # Occasional large price spikes
    shocks = np.zeros(n_days)
    n_shocks = 12  # Number of major events over 4 years
    shock_indices = np.random.choice(n_days, n_shocks, replace=False)
    for idx in shock_indices:
        shock_magnitude = np.random.uniform(8, 20)  # Large price increase
        shock_duration = np.random.randint(10, 40)  # Days
        decay = np.exp(-np.arange(shock_duration) / 10)  # Exponential decay
        end_idx = min(idx + shock_duration, n_days)
        shocks[idx:end_idx] += shock_magnitude * decay[:end_idx-idx]
    
    # 5. Autoregressive noise (price persistence)
    ar_noise = np.zeros(n_days)
    ar_noise[0] = np.random.normal(0, 1)
    for i in range(1, n_days):
        ar_noise[i] = 0.7 * ar_noise[i-1] + np.random.normal(0, 1.5)
    
    # Combine all components
    price = base_price + trend + annual_cycle + monthly_cycle + shocks + ar_noise
    
    # Ensure prices are positive and realistic
    price = np.maximum(price, 10)  # Minimum price of 10 rupees/kg
    
    # Generate volume data (inversely correlated with price)
    base_volume = 5000  # Quintals
    volume = base_volume - 50 * (price - base_price) + np.random.normal(0, 300, n_days)
    volume = np.maximum(volume, 1000)  # Minimum volume
    
    # Generate promotional flags (random 10% of days)
    promo_flag = np.random.choice([0, 1], size=n_days, p=[0.9, 0.1])
    
    # 6. Apply promotional discount (5-15% price reduction)
    promo_discount = np.where(promo_flag == 1, 
                              np.random.uniform(0.85, 0.95, n_days), 
                              1.0)
    price = price * promo_discount
    
    # Increase volume on promo days
    volume = np.where(promo_flag == 1, volume * 1.3, volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'price_rupees_per_kg': np.round(price, 2),
        'volume_quintals': np.round(volume, 2),
        'promo_flag': promo_flag
    })
    
    # Add temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    return df


def save_dataset(df, filepath='data/onion_prices_synthetic.csv'):
    """
    Save the generated dataset to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to save
    filepath : str
        Path where to save the CSV file
    """
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset statistics:")
    print(df.describe())


if __name__ == "__main__":
    # Generate the dataset
    print("Generating synthetic onion price dataset...")
    print("=" * 60)
    
    df = generate_onion_price_data()
    
    # Save to CSV
    save_dataset(df, 'data/onion_prices_synthetic.csv')
    
    print("\n" + "=" * 60)
    print("Data Generation Complete!")
    print("\nDataset Period: 2022-2025 (Latest 4 years)")
    print("Use this data to forecast 2026 prices!")
    print("\nDataset Components:")
    print("1. Long-term upward trend (inflation ~6% per year)")
    print("2. Annual seasonality (harvest cycles)")
    print("3. Monthly seasonality (within-year variations)")
    print("4. Random shocks (weather events, policy changes)")
    print("5. Autoregressive noise (price persistence)")
    print("6. Promotional effects (price discounts)")
