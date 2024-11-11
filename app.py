import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import time

# Set the title of the web app
st.set_page_config(page_title="Product Recommendations", layout="wide")

# Load the recommendations CSV file
df = pd.read_csv("recommendations.csv")

# Add a header and a subheader
st.title("Product Recommendations")
st.subheader("Find the best deals tailored for you!")

# Add a sidebar with additional options
st.sidebar.header("Options")
st.sidebar.write("Use the options below to customize your experience.")

# Sidebar for autorefresh settings
refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [0, 30, 60, 120])
if refresh_interval > 0:
    st_autorefresh(interval=refresh_interval * 1000, key="dataframe_refresh")

# Show the current time
st.sidebar.write("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Load data and display it
st.write("### Recommendations Data")
st.dataframe(df.style.highlight_max(axis=0), height=500)

# Create a button to refresh data
if st.button("Refresh Data"):
    df = pd.read_csv("recommendations.csv")
    st.experimental_rerun()

# Adding some custom CSS for styling
st.markdown("""
    <style>
        .reportview-container {
            background: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Footer
st.write("---")
st.write("© 2024 Product Recommendations. All rights reserved.")
