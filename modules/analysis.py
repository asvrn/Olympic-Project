import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def perform_analysis(dataframes):
    st.title("Olympic Games Analysis System")
    
    if 'medals' in dataframes:
        medals_df = dataframes['medals'].copy()
        
        # Pilihan jenis analisis
        analysis_type = st.sidebar.radio(
            "Choose Analysis Type",
            ["Simple Clustering Analysis", "Detailed Performance Analysis"]
        )
        
        if analysis_type == "Simple Clustering Analysis":
            perform_simple_analysis(medals_df)
        else:
            perform_detailed_analysis(medals_df)

def perform_simple_analysis(medals_df):
    st.header("Simple Clustering Analysis")
    
    # Slider untuk jumlah cluster
    n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    
    # Prepare data
    gold_counts = medals_df[medals_df['medal_type'] == 'GOLD'].groupby('country_name').size()
    silver_counts = medals_df[medals_df['medal_type'] == 'SILVER'].groupby('country_name').size()
    bronze_counts = medals_df[medals_df['medal_type'] == 'BRONZE'].groupby('country_name').size()
    
    country_medals = pd.DataFrame({
        'gold_count': gold_counts,
        'silver_count': silver_counts,
        'bronze_count': bronze_counts
    }).fillna(0)
    
    country_medals['total_medals'] = country_medals.sum(axis=1)
    country_medals = country_medals[country_medals['total_medals'] > 0]
    
    # Clustering
    features = ['gold_count', 'silver_count', 'bronze_count']
    X = country_medals[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    country_medals['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualisasi
    fig = px.scatter_3d(
        country_medals.reset_index(),
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
    
    # Analisis cluster
    for i in range(n_clusters):
        cluster_data = country_medals[country_medals['Cluster'] == i]
        st.subheader(f"Cluster {i} Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Top 5 Countries:")
            st.dataframe(
                cluster_data.nlargest(5, 'total_medals')[
                    ['gold_count', 'silver_count', 'bronze_count', 'total_medals']
                ]
            )
        
        with col2:
            st.write("Cluster Statistics:")
            st.write(f"Number of Countries: {len(cluster_data)}")
            st.write(f"Average Total Medals: {cluster_data['total_medals'].mean():.2f}")
            st.write(f"Average Gold Medals: {cluster_data['gold_count'].mean():.2f}")
    
    # Tambahan visualisasi karakteristik cluster
    st.subheader("Cluster Characteristics")
    
    # Hitung rata-rata medali per cluster
    cluster_means = country_medals.groupby('Cluster')[['gold_count', 'silver_count', 'bronze_count']].mean()
    
    # Buat visualisasi karakteristik cluster
    fig = go.Figure()
    
    # Tambahkan bar untuk setiap jenis medali
    fig.add_trace(go.Bar(
        name='Gold',
        x=cluster_means.index,
        y=cluster_means['gold_count'],
        text=cluster_means['gold_count'].round(1),
        textposition='auto',
        marker_color='gold',
    ))
    
    fig.add_trace(go.Bar(
        name='Silver',
        x=cluster_means.index,
        y=cluster_means['silver_count'],
        text=cluster_means['silver_count'].round(1),
        textposition='auto',
        marker_color='silver',
    ))
    
    fig.add_trace(go.Bar(
        name='Bronze',
        x=cluster_means.index,
        y=cluster_means['bronze_count'],
        text=cluster_means['bronze_count'].round(1),
        textposition='auto',
        marker_color='#cd7f32',
    ))
    
    fig.update_layout(
        barmode='group',
        title='Average Medal Counts per Cluster',
        xaxis_title='Cluster',
        yaxis_title='Average Number of Medals',
        template='plotly_dark'  # Menggunakan tema gelap seperti pada gambar
    )
    
    st.plotly_chart(fig)

def perform_detailed_analysis(medals_df):
    st.header("Detailed Performance Analysis")
    
    # Process data
    country_data = process_olympic_data(medals_df)
    
    # Sidebar untuk memilih cluster
    cluster_type = st.sidebar.selectbox(
        "Select Cluster Type",
        ["All Clusters", "Dominant Nations", "Developing Nations", "Specialist Nations", "Potential Nations"]
    )
    
    # Create and analyze clusters
    clustered_data = create_detailed_clusters(country_data)
    
    if cluster_type == "All Clusters":
        show_all_detailed_clusters(clustered_data)
    else:
        show_specific_detailed_cluster(clustered_data, medals_df, cluster_type)

def process_olympic_data(medals_df):
    # Calculate metrics
    medals_by_country = medals_df.groupby('country_name').agg({
        'medal_type': ['count', 
                      lambda x: (x == 'GOLD').sum(),
                      lambda x: (x == 'SILVER').sum(),
                      lambda x: (x == 'BRONZE').sum()],
        'discipline_title': 'nunique'
    })
    
    medals_by_country.columns = ['total_medals', 'gold_medals', 'silver_medals', 'bronze_medals', 'discipline_count']
    
    # Calculate ratios
    medals_by_country['gold_ratio'] = medals_by_country['gold_medals'] / medals_by_country['total_medals']
    medals_by_country['bronze_ratio'] = medals_by_country['bronze_medals'] / medals_by_country['total_medals']
    
    # Calculate specialization
    discipline_pivot = pd.crosstab(medals_df['country_name'], medals_df['discipline_title'])
    discipline_ratios = discipline_pivot.div(discipline_pivot.sum(axis=1), axis=0)
    medals_by_country['specialization_score'] = discipline_ratios.max(axis=1)
    
    # Get top disciplines
    top_disciplines = {}
    for country in medals_by_country.index:
        country_medals = medals_df[medals_df['country_name'] == country]
        top_discipline = country_medals['discipline_title'].mode().iloc[0]
        top_disciplines[country] = top_discipline
    
    medals_by_country['top_discipline'] = pd.Series(top_disciplines)
    
    return medals_by_country

def create_detailed_clusters(data):
    data['cluster_type'] = data.apply(determine_cluster_type, axis=1)
    return data

def determine_cluster_type(row):
    if row['gold_ratio'] > 0.35:
        return "Dominant Nations"
    elif row['specialization_score'] > 0.5:
        return "Specialist Nations"
    elif row['bronze_ratio'] > 0.5:
        return "Developing Nations"
    else:
        return "Potential Nations"

def show_all_detailed_clusters(data):
    st.subheader("Overview of All Clusters")
    
    # Create 3D visualization
    fig = px.scatter_3d(
        data.reset_index(),
        x='gold_ratio',
        y='bronze_ratio',
        z='specialization_score',
        color='cluster_type',
        hover_data=['country_name', 'total_medals', 'top_discipline'],
        title='Olympic Nations Performance Clusters',
        labels={
            'gold_ratio': 'Gold Medal Ratio',
            'bronze_ratio': 'Bronze Medal Ratio',
            'specialization_score': 'Specialization Score'
        }
    )
    st.plotly_chart(fig)
    
    # Show summaries
    cluster_types = ["Dominant Nations", "Developing Nations", "Specialist Nations", "Potential Nations"]
    for ct in cluster_types:
        show_cluster_summary(data, ct)

def show_specific_detailed_cluster(data, medals_df, cluster_type):
    cluster_data = data[data['cluster_type'] == cluster_type].copy()
    
    st.subheader(f"Analysis: {cluster_type}")
    
    # Display characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Cluster Statistics")
        st.write(f"Number of Countries: {len(cluster_data)}")
        st.write(f"Average Gold Ratio: {cluster_data['gold_ratio'].mean():.2%}")
        st.write(f"Average Total Medals: {cluster_data['total_medals'].mean():.1f}")
    
    with col2:
        if cluster_type == "Dominant Nations":
            st.write("💪 These nations consistently win gold medals across multiple disciplines")
            st.metric("Total Gold Medals", int(cluster_data['gold_medals'].sum()))
        elif cluster_type == "Developing Nations":
            st.write("🌱 These nations are building their Olympic programs")
            st.metric("Total Bronze Medals", int(cluster_data['bronze_medals'].sum()))
        elif cluster_type == "Specialist Nations":
            st.write("🎯 These nations excel in specific disciplines")
            st.write("Top Specializations:")
            st.write(cluster_data['top_discipline'].value_counts().head())
        else:
            st.write("⭐ These nations show promising performance trends")
            st.metric("Average Specialization", f"{cluster_data['specialization_score'].mean():.2f}")
    
    # Show country details
    st.subheader("Countries in this Cluster")
    country_details = cluster_data.reset_index()
    display_columns = ['country_name', 'total_medals', 'gold_medals', 'silver_medals', 'bronze_medals']
    
    if cluster_type == "Specialist Nations":
        display_columns.append('top_discipline')
    
    st.dataframe(
        country_details[display_columns]
        .sort_values('total_medals', ascending=False)
        .style.format({
            'total_medals': '{:.0f}',
            'gold_medals': '{:.0f}',
            'silver_medals': '{:.0f}',
            'bronze_medals': '{:.0f}'
        })
    )
    
    # Create visualization
    create_cluster_specific_viz(cluster_data, cluster_type)

def create_cluster_specific_viz(data, cluster_type):
    if cluster_type == "Dominant Nations":
        fig = px.bar(
            data.reset_index(),
            x='country_name',
            y=['gold_medals', 'silver_medals', 'bronze_medals'],
            title='Medal Distribution in Dominant Nations',
            barmode='stack'
        )
    elif cluster_type == "Developing Nations":
        fig = px.scatter(
            data.reset_index(),
            x='bronze_ratio',
            y='total_medals',
            size='discipline_count',
            hover_data=['country_name'],
            title='Development Progress by Nation'
        )
    elif cluster_type == "Specialist Nations":
        fig = px.bar(
            data.reset_index(),
            x='country_name',
            y='specialization_score',
            color='top_discipline',
            title='Specialization Scores by Country'
        )
    else:
        fig = px.scatter(
            data.reset_index(),
            x='gold_ratio',
            y='total_medals',
            size='discipline_count',
            hover_data=['country_name'],
            title='Growth Potential Indicators'
        )
    
    st.plotly_chart(fig)

def show_cluster_summary(data, cluster_type):
    cluster_data = data[data['cluster_type'] == cluster_type]
    
    st.subheader(f"{cluster_type} Summary")
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Countries", len(cluster_data))
    with cols[1]:
        st.metric("Avg Gold Ratio", f"{cluster_data['gold_ratio'].mean():.1%}")
    with cols[2]:
        st.metric("Avg Total Medals", f"{cluster_data['total_medals'].mean():.1f}")