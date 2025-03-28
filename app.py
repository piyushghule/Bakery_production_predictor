import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

from utils.data_processing import preprocess_data, validate_data
from utils.forecasting import train_forecast_model, make_predictions
from utils.visualization import (
    plot_sales_trends, 
    plot_product_distribution, 
    plot_sales_forecast,
    plot_seasonality
)
from utils.recommendations import generate_production_recommendations

# Set page config
st.set_page_config(
    page_title="Bakery Sales Forecasting",
    page_icon="🥐",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'forecast_model' not in st.session_state:
    st.session_state.forecast_model = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Function to reset app state
def reset_app_state():
    st.session_state.data = None
    st.session_state.forecast_model = None
    st.session_state.forecast_data = None
    st.session_state.recommendations = None

# App header
st.title("Bakery Sales Forecasting System")
st.markdown("### Optimize production and reduce waste with predictive insights")

# Sidebar for navigation and controls
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Data Upload", "Data Exploration", "Sales Forecasting", "Production Recommendations"]
)

# Data Upload Page
if page == "Data Upload":
    st.header("Upload Your Bakery Sales Data")
    st.markdown("""
    Upload your historical sales data to start forecasting. The data should include:
    - Date (date or datetime format)
    - Item type/name (string)
    - Quantity sold (numeric)
    - Revenue (numeric)
    - Cost of goods sold (COGS) (numeric)
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Read the file based on its extension
            file_extension = uploaded_file.name.split(".")[-1]
            if file_extension.lower() == "csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() == "xlsx":
                data = pd.read_excel(uploaded_file)
            
            # Validate data
            validation_result, message = validate_data(data)
            
            if validation_result:
                # Preprocess data
                processed_data = preprocess_data(data)
                st.session_state.data = processed_data
                
                # Display success message and data preview
                st.success("Data uploaded and processed successfully!")
                st.subheader("Data Preview")
                st.dataframe(processed_data.head())
                
                # Display basic statistics
                st.subheader("Basic Statistics")
                st.write(processed_data.describe())
            else:
                st.error(f"Error in data validation: {message}")
                st.info("Please ensure your data includes date, item name, quantity, revenue, and COGS columns.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please check your file format and try again.")
    
    # Option to use a reset button
    if st.session_state.data is not None:
        if st.button("Reset Data"):
            reset_app_state()
            st.success("Data has been reset. You can upload a new file.")
            st.rerun()

# Data Exploration Page
elif page == "Data Exploration":
    st.header("Data Exploration and Visualization")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Time range selector
        st.subheader("Select Time Range")
        min_date = data['date'].min().date()
        max_date = data['date'].max().date()
        date_range = st.date_input(
            "Date range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = data[(data['date'].dt.date >= start_date) & 
                                 (data['date'].dt.date <= end_date)]
            
            # Show visualizations in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sales Trends Over Time")
                sales_trend_fig = plot_sales_trends(filtered_data)
                st.plotly_chart(sales_trend_fig, use_container_width=True)
            
            with col2:
                st.subheader("Product Distribution")
                product_dist_fig = plot_product_distribution(filtered_data)
                st.plotly_chart(product_dist_fig, use_container_width=True)
            
            # Additional insights
            st.subheader("Key Insights")
            
            # Calculate metrics
            total_sales = filtered_data['revenue'].sum()
            total_items_sold = filtered_data['quantity'].sum()
            avg_daily_sales = filtered_data.groupby(filtered_data['date'].dt.date)['revenue'].sum().mean()
            best_selling_product = filtered_data.groupby('item')['quantity'].sum().idxmax()
            profit_margin = ((filtered_data['revenue'].sum() - filtered_data['cogs'].sum()) / 
                             filtered_data['revenue'].sum() * 100)
            
            # Display metrics in columns
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Total Sales", f"${total_sales:.2f}")
                st.metric("Total Items Sold", f"{total_items_sold}")
            
            with metrics_col2:
                st.metric("Avg. Daily Sales", f"${avg_daily_sales:.2f}")
                st.metric("Best Selling Product", best_selling_product)
            
            with metrics_col3:
                st.metric("Overall Profit Margin", f"{profit_margin:.2f}%")
                
            # Seasonality analysis
            st.subheader("Seasonality Analysis")
            seasonality_fig = plot_seasonality(filtered_data)
            st.plotly_chart(seasonality_fig, use_container_width=True)
            
    else:
        st.warning("Please upload your sales data first in the 'Data Upload' page.")
        if st.button("Go to Data Upload"):
            st.session_state.current_page = "Data Upload"
            st.rerun()

# Sales Forecasting Page
elif page == "Sales Forecasting":
    st.header("Sales Forecasting")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        st.subheader("Forecast Settings")
        
        # Forecast parameters
        forecast_period = st.slider("Forecast Period (days)", 7, 90, 30)
        
        # Option for selecting specific products or all products
        all_products = ["All Products"] + list(data['item'].unique())
        selected_product = st.selectbox("Select Product", all_products)
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            seasonality_mode = st.radio("Seasonality Mode", ["additive", "multiplicative"], index=0)
            changepoint_prior_scale = st.slider("Changepoint Prior Scale (flexibility of trend)", 
                                               0.001, 0.5, 0.05, step=0.001, format="%.3f")
            seasonality_prior_scale = st.slider("Seasonality Prior Scale", 
                                               0.01, 10.0, 1.0, step=0.01, format="%.2f")
        
        # Filter data based on selected product
        if selected_product != "All Products":
            forecast_data = data[data['item'] == selected_product].copy()
        else:
            # For all products, aggregate the data by date
            forecast_data = data.groupby('date').agg({
                'quantity': 'sum',
                'revenue': 'sum',
                'cogs': 'sum'
            }).reset_index()
        
        # Check if we have enough data for forecasting
        if len(forecast_data) < 14:  # Arbitrary minimum threshold
            st.warning("Not enough data for reliable forecasting. Please ensure you have at least 2 weeks of daily data.")
        else:
            if st.button("Generate Forecast"):
                with st.spinner("Training forecasting model..."):
                    # Train the model
                    model = train_forecast_model(
                        forecast_data, 
                        seasonality_mode=seasonality_mode,
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale
                    )
                    
                    # Make predictions
                    forecast = make_predictions(model, forecast_period)
                    
                    # Store results in session state
                    st.session_state.forecast_model = model
                    st.session_state.forecast_data = forecast
                
                # Display forecast results
                st.subheader("Sales Forecast Results")
                
                # Create forecast visualization
                forecast_fig = plot_sales_forecast(
                    historical_data=forecast_data, 
                    forecast_data=st.session_state.forecast_data,
                    product_name=selected_product
                )
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Display forecast statistics
                st.subheader("Forecast Statistics")
                
                # Calculate forecast metrics
                current_avg = forecast_data['quantity'].mean()
                forecast_avg = st.session_state.forecast_data['yhat'].mean()
                percent_change = ((forecast_avg - current_avg) / current_avg) * 100
                
                # Show metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Current Avg. Daily Sales", f"{current_avg:.2f} units")
                    st.metric("Peak Forecast Day", 
                             f"{st.session_state.forecast_data['yhat'].max():.2f} units",
                             f"Date: {st.session_state.forecast_data['ds'][st.session_state.forecast_data['yhat'].idxmax()].strftime('%Y-%m-%d')}")
                
                with metrics_col2:
                    st.metric("Forecast Avg. Daily Sales", 
                             f"{forecast_avg:.2f} units", 
                             f"{percent_change:.2f}%")
                    st.metric("Min Forecast Day", 
                             f"{st.session_state.forecast_data['yhat'].min():.2f} units",
                             f"Date: {st.session_state.forecast_data['ds'][st.session_state.forecast_data['yhat'].idxmin()].strftime('%Y-%m-%d')}")
                
                # Show forecast data in tabular form
                with st.expander("View Detailed Forecast Data"):
                    display_forecast = st.session_state.forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    display_forecast.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                    display_forecast['Date'] = display_forecast['Date'].dt.strftime('%Y-%m-%d')
                    display_forecast = display_forecast.round(2)
                    st.dataframe(display_forecast)
                    
                # Download forecast as CSV
                csv = display_forecast.to_csv(index=False)
                st.download_button(
                    label="Download Forecast as CSV",
                    data=csv,
                    file_name=f"bakery_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Generate recommendations based on forecast
                if st.session_state.forecast_data is not None:
                    st.session_state.recommendations = generate_production_recommendations(
                        st.session_state.forecast_data,
                        product_name=selected_product
                    )
    else:
        st.warning("Please upload your sales data first in the 'Data Upload' page.")
        if st.button("Go to Data Upload"):
            st.rerun()

# Production Recommendations Page
elif page == "Production Recommendations":
    st.header("Production Optimization Recommendations")
    
    if st.session_state.recommendations is not None:
        recommendations = st.session_state.recommendations
        
        st.subheader("Daily Production Plan")
        st.dataframe(recommendations['daily_plan'])
        
        # Production metrics
        st.subheader("Production Insights")
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Average Daily Production", f"{recommendations['avg_daily_production']:.2f} units")
            st.metric("Total Production (Forecast Period)", f"{recommendations['total_production']:.2f} units")
        
        with metrics_col2:
            st.metric("Peak Production Day", 
                     f"{recommendations['peak_production']:.2f} units",
                     f"Date: {recommendations['peak_production_date']}")
            st.metric("Production Buffer Applied", f"{recommendations['buffer_percentage']}%")
        
        # Risk assessment
        st.subheader("Risk Assessment")
        st.write(recommendations['risk_assessment'])
        
        # High risk days
        if len(recommendations['high_risk_days']) > 0:
            st.warning("**High Risk Days**")
            st.dataframe(recommendations['high_risk_days'])
        
        # Download production plan
        csv = recommendations['daily_plan'].to_csv(index=False)
        st.download_button(
            label="Download Production Plan as CSV",
            data=csv,
            file_name=f"bakery_production_plan_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Additional recommendations
        st.subheader("Additional Recommendations")
        st.write(recommendations['additional_recommendations'])
        
    else:
        if st.session_state.data is not None:
            st.info("Please generate a forecast first in the 'Sales Forecasting' page to see production recommendations.")
            if st.button("Go to Sales Forecasting"):
                st.rerun()
        else:
            st.warning("Please upload your sales data first in the 'Data Upload' page.")
            if st.button("Go to Data Upload"):
                st.rerun()

# Footer
st.markdown("---")
st.markdown("Bakery Sales Forecasting System | Helping bakers optimize production and reduce waste")
