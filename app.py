import streamlit as st
import traceback
import os

print("✅ Reached top of app.py")

st.set_page_config(page_title="Bakery Forecasting", layout="wide")

try:
    st.title("Bakery Forecasting App")
    st.text("✅ Basic Streamlit app loaded")

except Exception as e:
    st.error("🚨 Streamlit app crashed")
    st.code(traceback.format_exc())
    print(traceback.format_exc())
