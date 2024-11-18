import streamlit as st
import pandas as pd
import numpy as np

def load_dataset():
    st.title("Input Dataset ke Sistem")
    
    uploaded_files = {
        'medals': st.file_uploader("Upload medals dataset (CSV)", type=['csv']),
        'athletes': st.file_uploader("Upload athletes dataset (CSV)", type=['csv']),
        'results': st.file_uploader("Upload results dataset (CSV)", type=['csv']),
        'hosts': st.file_uploader("Upload hosts dataset (CSV)", type=['csv'])
    }
    
    dataframes = {}
    if any(uploaded_files.values()):
        st.header("Preview Data")
        for key, file in uploaded_files.items():
            if file is not None:
                try:
                    df = pd.read_csv(file)
                    dataframes[key] = df
                    
                    st.subheader(f"{key.capitalize()} Dataset")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Top 5 Rows:")
                        st.dataframe(df.head())
                    with col2:
                        st.write("Bottom 5 Rows:")
                        st.dataframe(df.tail())
                    
                    st.write(f"Shape: {df.shape}")
                    
                except Exception as e:
                    st.error(f"Error loading {key} dataset: {e}")
    
    return dataframes