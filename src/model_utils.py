"""
Model Utilities Module

This module contains utility functions for model training, evaluation,
and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing RMSE, MAE, and R2 score
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def plot_predictions(y_true, y_pred, dates=None, title='Actual vs Predicted Prices', 
                     figsize=(14, 6)):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    dates : array-like, optional
        Date values for x-axis
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    if dates is not None:
        plt.plot(dates, y_true, label='Actual', linewidth=2, alpha=0.7)
        plt.plot(dates, y_pred, label='Predicted', linewidth=2, alpha=0.7)
        plt.xlabel('Date', fontsize=12)
    else:
        plt.plot(y_true, label='Actual', linewidth=2, alpha=0.7)
        plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
        plt.xlabel('Time', fontsize=12)
    
    plt.ylabel('Price (Rupees/kg)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_residuals(y_true, y_pred, figsize=(14, 5)):
    """
    Plot residual analysis.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    figsize : tuple
        Figure size
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15, figsize=(10, 8)):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    return plt.gcf()


def save_model(model, filepath):
    """
    Save model to file using pickle.
    
    Parameters:
    -----------
    model : object
        Model to save
    filepath : str
        Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {filepath}")


def load_model(filepath):
    """
    Load model from file.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    object
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from: {filepath}")
    return model


def print_metrics(metrics, model_name="Model"):
    """
    Print metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    model_name : str
        Name of the model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        print(f"{metric_name:15s}: {value:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("Model Utilities Module")
    print("This module provides functions for:")
    print("- Calculating metrics (RMSE, MAE, R2)")
    print("- Plotting predictions and residuals")
    print("- Plotting feature importance")
    print("- Saving and loading models")
