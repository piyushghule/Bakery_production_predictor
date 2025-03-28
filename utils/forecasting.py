import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta

def train_forecast_model(data, seasonality_mode='additive', 
                         changepoint_prior_scale=0.05, 
                         seasonality_prior_scale=10.0):
    """
    Train a Prophet forecasting model on the provided data.
    
    Parameters:
    - data: Pandas DataFrame with at least 'date' and 'quantity' columns
    - seasonality_mode: 'additive' or 'multiplicative'
    - changepoint_prior_scale: Controls flexibility of trend (higher = more flexible)
    - seasonality_prior_scale: Controls flexibility of seasonality
    
    Returns:
    - model: Trained Prophet model
    """
    # Prepare data in Prophet format
    prophet_data = data[['date', 'quantity']].copy()
    prophet_data.columns = ['ds', 'y']
    
    # Remove any duplicate dates by aggregating
    prophet_data = prophet_data.groupby('ds')['y'].sum().reset_index()
    
    # Configure and train Prophet model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        daily_seasonality=True  # Important for bakery data which often has daily patterns
    )
    
    # Add weekly and yearly seasonality
    model.add_seasonality(name='weekly', period=7, fourier_order=3)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    
    # Fit the model
    model.fit(prophet_data)
    
    return model

def make_predictions(model, periods=30):
    """
    Generate forecasts using the trained model.
    
    Parameters:
    - model: Trained Prophet model
    - periods: Number of days to forecast
    
    Returns:
    - forecast: DataFrame with forecast results
    """
    # Create future dataframe for predictions
    future = model.make_future_dataframe(periods=periods)
    
    # Generate forecast
    forecast = model.predict(future)
    
    return forecast

def evaluate_forecast_accuracy(model, historical_data):
    """
    Evaluate the forecast model's accuracy on historical data.
    
    Parameters:
    - model: Trained Prophet model
    - historical_data: DataFrame with actual historical data
    
    Returns:
    - metrics: Dictionary of evaluation metrics
    """
    # Convert data to Prophet format
    prophet_data = historical_data[['date', 'quantity']].copy()
    prophet_data.columns = ['ds', 'y']
    
    # Make predictions on the historical data
    predictions = model.predict(pd.DataFrame({'ds': prophet_data['ds']}))
    
    # Calculate metrics
    actual = prophet_data['y'].values
    predicted = predictions['yhat'].values[:len(actual)]
    
    # Calculate error metrics
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Return metrics as a dictionary
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    return metrics
