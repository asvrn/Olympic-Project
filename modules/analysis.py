import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

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
    
    # Prepare data
    country_medals = prepare_data(medals_df)
    X_scaled = prepare_features(country_medals)
    
    # Silhouette Analysis
    st.subheader("Clustering Evaluation using Silhouette Analysis")
    optimal_k = perform_silhouette_analysis(X_scaled, country_medals)
    
    # User can still adjust number of clusters
    n_clusters = st.slider("Select number of clusters", 2, 10, optimal_k, 
                          help="Default value is the optimal number based on Silhouette analysis, but you can adjust it based on your needs")
    
    # Perform clustering with selected k
    perform_clustering_analysis(X_scaled, country_medals, n_clusters)

def prepare_data(medals_df):
    # Calculate medal counts per country
    gold_counts = medals_df[medals_df['medal_type'] == 'GOLD'].groupby('country_name').size()
    silver_counts = medals_df[medals_df['medal_type'] == 'SILVER'].groupby('country_name').size()
    bronze_counts = medals_df[medals_df['medal_type'] == 'BRONZE'].groupby('country_name').size()
    
    # Create DataFrame
    country_medals = pd.DataFrame({
        'gold_count': gold_counts,
        'silver_count': silver_counts,
        'bronze_count': bronze_counts
    }).fillna(0)
    
    country_medals['total_medals'] = country_medals.sum(axis=1)
    return country_medals[country_medals['total_medals'] > 0]

def prepare_features(country_medals):
    features = ['gold_count', 'silver_count', 'bronze_count']
    X = country_medals[features]
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def perform_silhouette_analysis(X_scaled, country_medals):
    # Calculate silhouette scores for different k
    k_range = range(2, 11)
    silhouette_scores = []
    silhouette_data = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        # Calculate silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
        silhouette_data.append({
            'k': k,
            'avg_score': silhouette_avg,
            'sample_values': sample_silhouette_values,
            'labels': cluster_labels
        })
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # Display results in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot silhouette scores
        fig1 = plot_silhouette_scores(k_range, silhouette_scores, optimal_k)
        st.plotly_chart(fig1)
        
        # Plot detailed silhouette for optimal k
        fig2 = plot_detailed_silhouette(silhouette_data, optimal_k)
        st.plotly_chart(fig2)
    
    with col2:
        st.info(f"""
        ### Optimal Clustering Results:
        - Number of clusters: **{optimal_k}**
        - Silhouette Score: **{max(silhouette_scores):.3f}**
        """)
        
        show_silhouette_interpretation(max(silhouette_scores))
        
        # Show all scores in a table
        st.write("### Silhouette Scores by K")
        scores_df = pd.DataFrame({
            'K': list(k_range),
            'Silhouette Score': silhouette_scores
        })
        st.dataframe(scores_df.style.format({'Silhouette Score': '{:.3f}'})
                    .background_gradient(subset=['Silhouette Score'], cmap='viridis'))
    
    return optimal_k

def plot_silhouette_scores(k_range, scores, optimal_k):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=scores,
        mode='lines+markers',
        name='Silhouette Score',
        line=dict(color='blue'),
    ))
    
    fig.add_trace(go.Scatter(
        x=[optimal_k],
        y=[scores[optimal_k-2]],
        mode='markers',
        marker=dict(size=15, symbol='star', color='red'),
        name=f'Optimal k={optimal_k}'
    ))
    
    fig.update_layout(
        title='Silhouette Scores by Number of Clusters',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Silhouette Score',
        template='plotly_dark'
    )
    
    return fig

def plot_detailed_silhouette(silhouette_data, optimal_k):
    optimal_data = next(data for data in silhouette_data if data['k'] == optimal_k)
    
    fig = go.Figure()
    
    y_lower = 0
    for i in range(optimal_k):
        cluster_values = optimal_data['sample_values'][optimal_data['labels'] == i]
        cluster_values.sort()
        
        y_upper = y_lower + len(cluster_values)
        
        fig.add_trace(go.Scatter(
            x=cluster_values,
            y=np.linspace(y_lower, y_upper, len(cluster_values)),
            mode='lines',
            name=f'Cluster {i}',
            showlegend=True
        ))
        
        y_lower = y_upper + 10
    
    fig.update_layout(
        title=f'Detailed Silhouette Plot for k={optimal_k}',
        xaxis_title='Silhouette Coefficient',
        yaxis_title='Cluster',
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def show_silhouette_interpretation(score):
    st.write("### Interpretation of Silhouette Score")
    st.write("""
    The silhouette score ranges from -1 to 1:
    - Score close to 1: Well-defined clusters
    - Score close to 0: Overlapping clusters
    - Score close to -1: Incorrect clustering
    """)
    
    if score > 0.7:
        st.success("Strong clustering structure found!")
    elif score > 0.5:
        st.info("Reasonable clustering structure detected")
    elif score > 0.25:
        st.warning("Weak clustering structure - consider different features or preprocessing")
    else:
        st.error("Poor clustering structure - data might not have clear clusters")

def perform_clustering_analysis(X_scaled, country_medals, n_clusters):
    st.header(f"Clustering Analysis with {n_clusters} clusters")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    country_medals['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Create 3D visualization
    fig = px.scatter_3d(
        country_medals.reset_index(),
        x='gold_count',
        y='silver_count',
        z='bronze_count',
        color='Cluster',
        hover_data=['country_name', 'total_medals'],
        title=f'Countries Clustered into {n_clusters} Groups',
        labels={
            'gold_count': 'Gold Medals',
            'silver_count': 'Silver Medals',
            'bronze_count': 'Bronze Medals'
        }
    )
    
    st.plotly_chart(fig)
    
    # Cluster characteristics visualization
    cluster_means = country_medals.groupby('Cluster')[['gold_count', 'silver_count', 'bronze_count']].mean()
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        name='Gold',
        x=cluster_means.index,
        y=cluster_means['gold_count'],
        marker_color='gold',
    ))
    
    fig2.add_trace(go.Bar(
        name='Silver',
        x=cluster_means.index,
        y=cluster_means['silver_count'],
        marker_color='silver',
    ))
    
    fig2.add_trace(go.Bar(
        name='Bronze',
        x=cluster_means.index,
        y=cluster_means['bronze_count'],
        marker_color='#cd7f32',
    ))
    
    fig2.update_layout(
        title='Average Medal Counts per Cluster',
        xaxis_title='Cluster',
        yaxis_title='Average Number of Medals',
        barmode='group',
        template='plotly_dark'
    )
    
    st.plotly_chart(fig2)
    
    # Display cluster analysis
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