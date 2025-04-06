import os
print("✅ DEBUG: Entered app.py")  # Will show in logs

import streamlit as st
import traceback

st.set_page_config(
    page_title="Bakery Sales Forecasting",
    page_icon="🥐",
    layout="wide"
)

st.text("✅ App has started...")
import streamlit as st
import traceback

st.set_page_config(
    page_title="Bakery Sales Forecasting",
    page_icon="🥐",
    layout="wide"
)

st.text("✅ App has started...")

try:
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

    # You can now begin adding back your app logic step-by-step below.
    # Use st.text("Loaded section XYZ") to track what works

except Exception as e:
    st.error("🚨 The app crashed due to an error during startup.")
    st.code(traceback.format_exc())