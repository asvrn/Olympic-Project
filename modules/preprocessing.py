import streamlit as st
import pandas as pd
import numpy as np

def check_column_names(df):
    """Helper function to find country and athlete related columns"""
    country_columns = [col for col in df.columns if 'country' in col.lower() or 'noc' in col.lower()]
    athlete_columns = [col for col in df.columns if 'athlete' in col.lower()]
    return {
        'country_columns': country_columns,
        'athlete_columns': athlete_columns
    }

def detailed_eda(dataframes):
    st.title("Preprocess Data di Sistem")
    
    processed_dfs = {}
    
    for name, df in dataframes.items():
        st.header(f"Analysis of {name.capitalize()} Dataset")
        
        # Basic Information
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Dataset Shape:", df.shape)
            st.write("Datatypes:")
            st.write(df.dtypes)
        with col2:
            st.write("Memory Usage:", df.memory_usage().sum() / 1024**2, "MB")
            st.write("Number of Duplicates:", df.duplicated().sum())
        
        # Missing Values Analysis
        st.subheader("Missing Values Analysis")
        missing = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing[missing['Missing Values'] > 0])
        
        # Handling Missing Values
        if missing['Missing Values'].sum() > 0:
            st.write("Handling Missing Values:")
            
            for column in df.columns[df.isnull().any()]:
                st.write(f"\nColumn: {column}")
                
                # Show unique values and their counts
                st.write("Value Distribution:")
                st.write(df[column].value_counts().head())
                
                # Suggest handling method based on missing percentage and data type
                missing_pct = (df[column].isnull().sum() / len(df)) * 100
                
                if missing_pct < 5:  # Low missing percentage
                    if df[column].dtype in ['int64', 'float64']:
                        df[column].fillna(df[column].mean(), inplace=True)
                        st.write("Action: Filled with mean value (low missing percentage)")
                    else:
                        df[column].fillna(df[column].mode()[0], inplace=True)
                        st.write("Action: Filled with mode value (low missing percentage)")
                else:  # High missing percentage
                    if column in ['athlete_full_name', 'country_name', 'medal_type']:
                        # Critical columns - keep null
                        st.write("Action: Kept null values (critical column)")
                    else:
                        df[column].fillna('Unknown', inplace=True)
                        st.write("Action: Filled with 'Unknown' (high missing percentage)")
        
        # Statistical Summary
        if df.select_dtypes(include=[np.number]).columns.any():
            st.subheader("Statistical Summary")
            st.write(df.describe())
        
        # Save processed dataframe
        processed_dfs[name] = df
    
    return processed_dfs