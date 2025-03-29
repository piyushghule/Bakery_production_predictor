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
    
    # Display all columns for debugging
    print(f"All columns in uploaded data: {list(data.columns)}")
    
    # Convert all column names to lowercase for easier matching
    data.columns = [col.lower().strip() for col in data.columns]
    
    # Check for required columns (flexible with column naming)
    required_columns = {
        'date': ['date', 'datetime', 'day', 'sale_date', 'transaction_date', 'date of sale', 'sales date'],
        'item': ['item', 'product', 'product_name', 'item_name', 'item_type', 'product type', 'goods', 'bakery item'],
        'quantity': ['quantity', 'qty', 'units', 'units_sold', 'quantity_sold', 'count', 'number sold', 'amount sold'],
        'revenue': ['revenue', 'sales', 'amount', 'sales_amount', 'income', 'sales revenue', 'total sales', 'price'],
        'cogs': ['cogs', 'cost', 'cost_of_goods_sold', 'costs', 'expense', 'expenses', 'production cost', 'cost price']
    }
    
    # Create a mapping from the actual column names to the standard names
    column_mapping = {}
    missing_columns = []
    found_columns = {}
    
    # First check: Find matches for required columns
    for std_name, possible_names in required_columns.items():
        found = False
        for col_name in possible_names:
            # Check for exact match
            if col_name in data.columns:
                column_mapping[col_name] = std_name
                found_columns[std_name] = col_name
                found = True
                break
                
            # Check for partial match (e.g., "product id" would match with "product")
            for col in data.columns:
                if col_name in col:
                    column_mapping[col] = std_name
                    found_columns[std_name] = col
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            missing_columns.append(std_name)
    
    # If columns are missing, provide detailed feedback
    if missing_columns:
        feedback = f"Missing required columns: {', '.join(missing_columns)}\n\n"
        feedback += "Your data needs to have columns that represent:\n"
        for missing in missing_columns:
            feedback += f"- {missing.capitalize()}: any of these names would work: {', '.join(required_columns[missing])}\n"
        feedback += "\nAvailable columns in your data: " + ", ".join(data.columns)
        return False, feedback
    
    # Print the mappings for debugging
    print(f"Column mapping: {column_mapping}")
    print(f"Found columns: {found_columns}")
    
    # Second check: Verify data types and handle conversion errors gracefully
    validation_errors = []
    
    # Validate date column
    try:
        date_col = found_columns['date']
        # Check for null values
        if data[date_col].isnull().any():
            null_count = data[date_col].isnull().sum()
            validation_errors.append(f"Date column '{date_col}' contains {null_count} missing values.")
        else:
            # Try to convert to datetime
            pd.to_datetime(data[date_col])
    except Exception as e:
        validation_errors.append(f"Error in date column '{date_col}': {str(e)}")
        validation_errors.append("The date column must contain valid dates (e.g., '2024-03-15', '03/15/2024').")
    
    # Validate numeric columns
    numeric_cols = {
        'quantity': found_columns['quantity'],
        'revenue': found_columns['revenue'],
        'cogs': found_columns['cogs']
    }
    
    for col_name, actual_col in numeric_cols.items():
        try:
            # Check for null values
            if data[actual_col].isnull().any():
                null_count = data[actual_col].isnull().sum()
                validation_errors.append(f"{col_name.capitalize()} column '{actual_col}' contains {null_count} missing values.")
            
            # Try to convert to numeric
            numeric_data = pd.to_numeric(data[actual_col], errors='coerce')
            
            # Check if conversion introduced NaN values
            if numeric_data.isnull().any():
                non_numeric_count = numeric_data.isnull().sum()
                non_numeric_rows = data.index[numeric_data.isnull()].tolist()[:5]  # Show first 5 problematic rows
                validation_errors.append(f"{col_name.capitalize()} column '{actual_col}' contains {non_numeric_count} non-numeric values.")
                validation_errors.append(f"Problematic rows (first 5): {non_numeric_rows}")
        except Exception as e:
            validation_errors.append(f"Error in {col_name} column '{actual_col}': {str(e)}")
    
    # Check item column for missing values
    try:
        item_col = found_columns['item']
        if data[item_col].isnull().any():
            null_count = data[item_col].isnull().sum()
            validation_errors.append(f"Item column '{item_col}' contains {null_count} missing values.")
    except Exception as e:
        validation_errors.append(f"Error in item column '{item_col}': {str(e)}")
    
    # If there are validation errors, return them
    if validation_errors:
        error_message = "Data validation failed with the following issues:\n" + "\n".join(validation_errors)
        return False, error_message
    
    # All validation passed
    return True, "Data validation successful! Your data has all required columns and formats."

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
    
    # Ensure column names are lowercase
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Expanded list of possible column names for better matching
    column_patterns = {
        'date': ['date', 'datetime', 'day', 'sale_date', 'transaction_date', 'date of sale', 'sales date'],
        'item': ['item', 'product', 'product_name', 'item_name', 'item_type', 'product type', 'goods', 'bakery item'],
        'quantity': ['quantity', 'qty', 'units', 'units_sold', 'quantity_sold', 'count', 'number sold', 'amount sold'],
        'revenue': ['revenue', 'sales', 'amount', 'sales_amount', 'income', 'sales revenue', 'total sales', 'price'],
        'cogs': ['cogs', 'cost', 'cost_of_goods_sold', 'costs', 'expense', 'expenses', 'production cost', 'cost price']
    }
    
    # Find the best match for each standard column
    column_mapping = {}
    
    for std_col, patterns in column_patterns.items():
        # First check for exact matches
        exact_matches = [col for col in df.columns if col in patterns]
        if exact_matches:
            column_mapping[std_col] = exact_matches[0]
            continue
            
        # Then check for partial matches
        for col in df.columns:
            for pattern in patterns:
                if pattern in col:
                    column_mapping[std_col] = col
                    break
            if std_col in column_mapping:
                break
    
    print(f"Found column mapping: {column_mapping}")
    
    # Rename columns to standard names
    rename_dict = {v: k for k, v in column_mapping.items()}
    df.rename(columns=rename_dict, inplace=True)
    
    # Ensure we have all required columns
    required_cols = ['date', 'item', 'quantity', 'revenue', 'cogs']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found after preprocessing. Please check your data format.")
    
    # Convert data types with more robust error handling
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop rows with invalid dates
        invalid_dates = df['date'].isnull().sum()
        if invalid_dates > 0:
            print(f"Dropping {invalid_dates} rows with invalid date values")
            df.dropna(subset=['date'], inplace=True)
    except Exception as e:
        print(f"Error converting date column: {str(e)}")
        # Attempt more aggressive date parsing if standard method fails
        try:
            df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
            df.dropna(subset=['date'], inplace=True)
        except:
            raise ValueError("Could not parse date column. Please ensure dates are in a standard format.")
    
    # Convert numeric columns
    for col in ['quantity', 'revenue', 'cogs']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
    
    # Print summary for debugging
    print(f"Preprocessed {len(df)} rows with columns: {list(df.columns)}")
    
    return df
