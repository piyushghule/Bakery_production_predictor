import pandas as pd
import numpy as np
from datetime import datetime

def validate_data(data):
    """
    Validate that the uploaded data contains the required columns and formats.
    
    Parameters:
    - data: Pandas DataFrame containing the uploaded data
    
    Returns:
    - validation_result: Boolean indicating if validation passed
    - message: String with error message if validation failed
    """
    # Check if DataFrame is empty
    if data.empty:
        return False, "The uploaded file contains no data."
    
    # Check for required columns (flexible with column naming)
    required_columns = {
        'date': ['date', 'datetime', 'day', 'sale_date', 'transaction_date'],
        'item': ['item', 'product', 'product_name', 'item_name', 'item_type'],
        'quantity': ['quantity', 'qty', 'units', 'units_sold', 'quantity_sold'],
        'revenue': ['revenue', 'sales', 'amount', 'sales_amount', 'income'],
        'cogs': ['cogs', 'cost', 'cost_of_goods_sold', 'costs', 'expense']
    }
    
    # Create a mapping from the actual column names to the standard names
    column_mapping = {}
    missing_columns = []
    
    for std_name, possible_names in required_columns.items():
        found = False
        for col_name in possible_names:
            if col_name in data.columns:
                column_mapping[col_name] = std_name
                found = True
                break
        
        if not found:
            missing_columns.append(std_name)
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check data types
    try:
        # Get the actual column name that maps to 'date'
        date_col = next(col for col, std in column_mapping.items() if std == 'date')
        # Try to convert to datetime
        pd.to_datetime(data[date_col])
    except Exception as e:
        return False, f"Error converting date column to datetime: {str(e)}"
    
    try:
        # Get the actual column names that map to numeric fields
        qty_col = next(col for col, std in column_mapping.items() if std == 'quantity')
        revenue_col = next(col for col, std in column_mapping.items() if std == 'revenue')
        cogs_col = next(col for col, std in column_mapping.items() if std == 'cogs')
        
        # Try to convert to numeric
        pd.to_numeric(data[qty_col])
        pd.to_numeric(data[revenue_col])
        pd.to_numeric(data[cogs_col])
    except Exception as e:
        return False, f"Error converting numeric columns: {str(e)}"
    
    return True, "Data validation successful"

def preprocess_data(data):
    """
    Preprocess the validated data for analysis and forecasting.
    
    Parameters:
    - data: Pandas DataFrame containing the validated data
    
    Returns:
    - processed_data: Pandas DataFrame with standardized columns and cleaned data
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Identify column names
    date_col = next((c for c in df.columns if c.lower() in ['date', 'datetime', 'day', 'sale_date', 'transaction_date']), None)
    item_col = next((c for c in df.columns if c.lower() in ['item', 'product', 'product_name', 'item_name', 'item_type']), None)
    qty_col = next((c for c in df.columns if c.lower() in ['quantity', 'qty', 'units', 'units_sold', 'quantity_sold']), None)
    revenue_col = next((c for c in df.columns if c.lower() in ['revenue', 'sales', 'amount', 'sales_amount', 'income']), None)
    cogs_col = next((c for c in df.columns if c.lower() in ['cogs', 'cost', 'cost_of_goods_sold', 'costs', 'expense']), None)
    
    # Standardize column names
    df.rename(columns={
        date_col: 'date',
        item_col: 'item',
        qty_col: 'quantity',
        revenue_col: 'revenue',
        cogs_col: 'cogs'
    }, inplace=True)
    
    # Convert data types
    df['date'] = pd.to_datetime(df['date'])
    df['quantity'] = pd.to_numeric(df['quantity'])
    df['revenue'] = pd.to_numeric(df['revenue'])
    df['cogs'] = pd.to_numeric(df['cogs'])
    
    # Handle missing values
    df.dropna(subset=['date', 'item', 'quantity'], inplace=True)
    df['revenue'].fillna(0, inplace=True)
    df['cogs'].fillna(0, inplace=True)
    
    # Add derived features
    df['profit'] = df['revenue'] - df['cogs']
    df['profit_margin'] = (df['profit'] / df['revenue']) * 100
    df['profit_margin'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['profit_margin'].fillna(0, inplace=True)
    
    # Add time-based features
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month_name()
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Sort by date
    df.sort_values('date', inplace=True)
    
    return df
