import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def plot_sales_trends(data):
    """
    Create a line chart of sales trends over time.
    
    Parameters:
    - data: Pandas DataFrame with 'date' and 'revenue' columns
    
    Returns:
    - fig: Plotly figure object
    """
    # Aggregate data by date
    daily_sales = data.groupby('date')['revenue'].sum().reset_index()
    
    # Create figure
    fig = px.line(
        daily_sales, 
        x='date', 
        y='revenue',
        labels={'date': 'Date', 'revenue': 'Revenue ($)'},
        title='Daily Sales Trend'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        hovermode='x unified'
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def plot_product_distribution(data):
    """
    Create a bar chart showing the distribution of products by quantity sold.
    
    Parameters:
    - data: Pandas DataFrame with 'item' and 'quantity' columns
    
    Returns:
    - fig: Plotly figure object
    """
    # Aggregate by product
    product_sales = data.groupby('item')['quantity'].sum().reset_index()
    product_sales = product_sales.sort_values('quantity', ascending=False)
    
    # Limit to top 10 products for better visualization
    if len(product_sales) > 10:
        top_products = product_sales.head(10)
        fig_title = 'Top 10 Products by Quantity Sold'
    else:
        top_products = product_sales
        fig_title = 'Products by Quantity Sold'
    
    # Create figure
    fig = px.bar(
        top_products,
        x='item',
        y='quantity',
        labels={'item': 'Product', 'quantity': 'Quantity Sold'},
        title=fig_title,
        color='quantity',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Product',
        yaxis_title='Quantity Sold',
        xaxis={'categoryorder': 'total descending'},
        coloraxis_showscale=False
    )
    
    return fig

def plot_sales_forecast(historical_data, forecast_data, product_name="All Products"):
    """
    Create a plot showing historical sales data and forecasted values.
    
    Parameters:
    - historical_data: DataFrame with historical sales data
    - forecast_data: DataFrame with forecasted values from Prophet
    - product_name: Name of the product being forecasted
    
    Returns:
    - fig: Plotly figure object
    """
    # Prepare historical data
    if 'date' in historical_data.columns:
        historical = historical_data[['date', 'quantity']].copy()
        historical.columns = ['ds', 'y']
    else:
        historical = historical_data[['ds', 'y']].copy()
    
    # Create a combined dataframe with historical and forecasted data
    historical['type'] = 'Historical'
    
    # For forecasted data, extract only future dates
    last_historical_date = historical['ds'].max()
    future_forecast = forecast_data[forecast_data['ds'] > last_historical_date].copy()
    future_forecast['type'] = 'Forecast'
    
    # Combine dataframes
    forecast_plot_data = pd.concat([
        historical,
        future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'type']]
    ])
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical['ds'],
        y=historical['y'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='royalblue')
    ))
    
    # Add forecasted values
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='firebrick')
    ))
    
    # Add prediction intervals
    fig.add_trace(go.Scatter(
        x=pd.concat([future_forecast['ds'], future_forecast['ds'][::-1]]),
        y=pd.concat([future_forecast['yhat_upper'], future_forecast['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Prediction Interval (95%)'
    ))
    
    # Customize layout
    fig.update_layout(
        title=f'Sales Forecast for {product_name}',
        xaxis_title='Date',
        yaxis_title='Quantity',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_seasonality(data):
    """
    Create a composite plot showing daily, weekly, and monthly seasonality patterns.
    
    Parameters:
    - data: Pandas DataFrame with sales data
    
    Returns:
    - fig: Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Sales by Day of Week', 'Monthly Sales Pattern', 
                        'Hourly Sales Pattern (if available)', 'Sales by Day of Month'),
        specs=[[{}, {}], [{}, {}]]
    )
    
    # Check if we have hourly data
    has_hourly_data = 'hour' in data.columns
    
    # 1. Sales by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_sales = data.groupby('day_of_week')['quantity'].mean().reindex(day_order).reset_index()
    
    fig.add_trace(
        go.Bar(
            x=weekly_sales['day_of_week'],
            y=weekly_sales['quantity'],
            marker_color='royalblue'
        ),
        row=1, col=1
    )
    
    # 2. Monthly sales pattern
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_sales = data.groupby('month')['quantity'].mean().reindex(month_order).reset_index()
    
    fig.add_trace(
        go.Bar(
            x=monthly_sales['month'],
            y=monthly_sales['quantity'],
            marker_color='firebrick'
        ),
        row=1, col=2
    )
    
    # 3. Hourly sales pattern (if available)
    if has_hourly_data:
        hourly_sales = data.groupby('hour')['quantity'].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=hourly_sales['hour'],
                y=hourly_sales['quantity'],
                mode='lines+markers',
                line=dict(color='green')
            ),
            row=2, col=1
        )
    else:
        # Alternative: Sales trend over the analysis period
        daily_trend = data.groupby('date')['quantity'].sum().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=daily_trend['date'],
                y=daily_trend['quantity'],
                mode='lines',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
    
    # 4. Sales by day of month
    day_of_month = data.groupby('day')['quantity'].mean().reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=day_of_month['day'],
            y=day_of_month['quantity'],
            mode='lines+markers',
            line=dict(color='purple')
        ),
        row=2, col=2
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Day of Week", row=1, col=1)
    fig.update_yaxes(title_text="Avg Quantity", row=1, col=1)
    
    fig.update_xaxes(title_text="Month", row=1, col=2)
    fig.update_yaxes(title_text="Avg Quantity", row=1, col=2)
    
    if has_hourly_data:
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    fig.update_yaxes(title_text="Avg Quantity", row=2, col=1)
    
    fig.update_xaxes(title_text="Day of Month", row=2, col=2)
    fig.update_yaxes(title_text="Avg Quantity", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text="Seasonality Analysis",
        showlegend=False
    )
    
    return fig
