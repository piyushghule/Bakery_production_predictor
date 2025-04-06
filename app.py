print("✅ Step 1: importing os")
import os

print("✅ Step 2: importing streamlit")
import streamlit as st

print("✅ Step 3: importing traceback")
import traceback

print("✅ Step 4: setting Streamlit config")
st.set_page_config(
    page_title="Bakery Sales Forecasting",
    page_icon="🥐",
    layout="wide"
)

st.text("✅ App has started...")
print("✅ Step 5: Streamlit display success")

try:
    print("🔄 Importing pandas, numpy, datetime, io...")
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import io

    print("🔄 Importing utils modules...")
    from utils.data_processing import preprocess_data, validate_data
    print("✅ data_processing imported")

    from utils.forecasting import train_forecast_model, make_predictions
    print("✅ forecasting imported")

    from utils.visualization import (
        plot_sales_trends,
        plot_product_distribution,
        plot_sales_forecast,
        plot_seasonality
    )
    print("✅ visualization imported")

    from utils.recommendations import generate_production_recommendations
    print("✅ recommendations imported")

    st.text("✅ All modules imported successfully.")

    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'forecast_model' not in st.session_state:
        st.session_state.forecast_model = None
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    st.title("Bakery Sales Forecasting System")
    st.markdown("### This is a debug-enabled version to trace errors during deployment")

    st.text("✅ Session state initialized.")

except Exception as e:
    st.error("🚨 The app crashed due to an error during startup.")
    st.code(traceback.format_exc())
    print("❌ ERROR during app startup:\n", traceback.format_exc())