import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def perform_analysis(dataframes):
    st.title("Analisis Data di Sistem")
    
    if 'medals' in dataframes:
        medals_df = dataframes['medals'].copy()
        
        st.header("Clustering Analysis of Olympic Medal Performance")
        
        # Prepare data for clustering
        # Calculate medal counts for each country separately
        gold_counts = medals_df[medals_df['medal_type'] == 'GOLD'].groupby('country_name').size()
        silver_counts = medals_df[medals_df['medal_type'] == 'SILVER'].groupby('country_name').size()
        bronze_counts = medals_df[medals_df['medal_type'] == 'BRONZE'].groupby('country_name').size()
        
        # Combine the counts into a single dataframe
        country_medals = pd.DataFrame({
            'gold_count': gold_counts,
            'silver_count': silver_counts,
            'bronze_count': bronze_counts
        }).fillna(0)
        
        # Reset index to make country a column
        country_medals = country_medals.reset_index()
        
        # Calculate total medals
        country_medals['total_medals'] = country_medals[['gold_count', 'silver_count', 'bronze_count']].sum(axis=1)
        
        # Remove countries with no medals
        country_medals = country_medals[country_medals['total_medals'] > 0]
        
        # Normalize the features
        features = ['gold_count', 'silver_count', 'bronze_count']
        X = country_medals[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        country_medals['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Visualize clusters
        st.subheader("3D Visualization of Country Clusters")
        fig = px.scatter_3d(
            country_medals,
            x='gold_count',
            y='silver_count',
            z='bronze_count',
            color='Cluster',
            hover_data=['country_name', 'total_medals'],
            title='Country Clusters based on Medal Performance',
            labels={
                'gold_count': 'Gold Medals',
                'silver_count': 'Silver Medals',
                'bronze_count': 'Bronze Medals'
            }
        )
        st.plotly_chart(fig)
        
        # Analyze clusters
        st.subheader("Cluster Analysis")
        for i in range(n_clusters):
            cluster_data = country_medals[country_medals['Cluster'] == i]
            
            st.write(f"\nCluster {i}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Top 5 Countries in Cluster:")
                st.dataframe(
                    cluster_data.nlargest(5, 'total_medals')[
                        ['country_name', 'gold_count', 'silver_count', 'bronze_count', 'total_medals']
                    ]
                )
            
            with col2:
                st.write("Cluster Statistics:")
                st.write(f"Number of Countries: {len(cluster_data)}")
                st.write(f"Average Total Medals: {cluster_data['total_medals'].mean():.2f}")
                st.write(f"Average Gold Medals: {cluster_data['gold_count'].mean():.2f}")
        
        # Add cluster characteristics visualization
        st.subheader("Cluster Characteristics")
        cluster_means = country_medals.groupby('Cluster')[features].mean()
        
        fig = go.Figure()
        for medal in features:
            fig.add_trace(go.Bar(
                name=medal.replace('_count', '').capitalize(),
                x=cluster_means.index,
                y=cluster_means[medal],
                text=cluster_means[medal].round(1),
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            title='Average Medal Counts per Cluster',
            xaxis_title='Cluster',
            yaxis_title='Average Number of Medals'
        )
        st.plotly_chart(fig)