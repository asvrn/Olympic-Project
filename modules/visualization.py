import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_visualization(dataframes):
    st.title("Visualisasi Data di Sistem")
    
    if 'medals' not in dataframes or 'hosts' not in dataframes:
        st.warning("Please upload medals and hosts datasets for visualization.")
        return
    
    medals_df = dataframes['medals'].copy()
    hosts_df = dataframes['hosts'].copy()
    
    # Initialize widget key
    widget_key = st.session_state.get('widget_key', 0)
    
    # Initialize default values
    DEFAULTS = {
        'season': 'All',
        'country': 'All',
        'discipline': 'All',
        'athlete': 'All'
    }
    
    # Initialize session state for filters
    if 'filters' not in st.session_state:
        st.session_state.filters = DEFAULTS.copy()
    
    # Create filters
    st.sidebar.header("FILTERS")
    
    # Season filter
    seasons = ['All'] + sorted(hosts_df['game_season'].unique().tolist())
    season = st.sidebar.selectbox(
        "Select Season",
        seasons,
        index=seasons.index(st.session_state.filters['season']),
        key=f'season_select_{widget_key}'
    )
    st.session_state.filters['season'] = season
    # Year range slider
    year_min = int(hosts_df['game_year'].min())
    year_max = int(hosts_df['game_year'].max())
    year_range = st.sidebar.slider(
        "Select Years",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max) if not 'year_range' in st.session_state.filters else st.session_state.filters['year_range'],
        key=f'year_slider_{widget_key}'
    )
    st.session_state.filters['year_range'] = year_range
    
    # Country filter
    countries = ['All'] + sorted(medals_df['country_name'].dropna().unique().tolist())
    country = st.sidebar.selectbox(
        "Select Country",
        countries,
        index=countries.index(st.session_state.filters['country']),
        key=f'country_select_{widget_key}'
    )
    st.session_state.filters['country'] = country
    
    # Discipline filter
    disciplines = ['All'] + sorted(medals_df['discipline_title'].dropna().unique().tolist())
    discipline = st.sidebar.selectbox(
        "Select Discipline",
        disciplines,
        index=disciplines.index(st.session_state.filters['discipline']),
        key=f'discipline_select_{widget_key}'
    )
    st.session_state.filters['discipline'] = discipline
    
    # Athlete filter
    athletes = ['All'] + sorted(medals_df['athlete_full_name'].dropna().unique().tolist())
    athlete = st.sidebar.selectbox(
        "Select Athlete",
        athletes,
        index=athletes.index(st.session_state.filters['athlete']),
        key=f'athlete_select_{widget_key}'
    )
    st.session_state.filters['athlete'] = athlete
    
    # Apply filters
    filtered_df = medals_df.copy()

    # Reset filters button
    reset_clicked = st.sidebar.button("Reset Filters", key='reset_button')
    if reset_clicked:
        st.session_state.filters = DEFAULTS.copy()
        if 'year' in filtered_df.columns:
            st.session_state.filters['year_range'] = (year_min, year_max)
        st.session_state.widget_key = st.session_state.get('widget_key', 0) + 1
        st.experimental_rerun()
    
    # Apply filters
    filtered_df = medals_df.copy()
    
    # Apply season filter
    if season != 'All':
    
        filtered_df = pd.merge(
            filtered_df,
            hosts_df[['game_slug', 'game_season']],  # Ambil kolom yang diperlukan dari hosts_df
            left_on='slug_game',                     # Kolom penghubung di medals_df
            right_on='game_slug',                    # Kolom penghubung di hosts_df
            how='left'
        )
        filtered_df = filtered_df[filtered_df['game_season'] == season]
    
    # Apply year filter (yang sudah ada sebelumnya)
    filtered_df = pd.merge(
        filtered_df,
        hosts_df[['game_slug', 'game_year']],
        left_on='slug_game',
        right_on='game_slug',
        how='left'
    )

    filtered_df = filtered_df[
        (filtered_df['game_year'] >= year_range[0]) &
        (filtered_df['game_year'] <= year_range[1])
    ]
    
    # Apply other filters
    if country != 'All':
        filtered_df = filtered_df[filtered_df['country_name'] == country]
    if discipline != 'All':
        filtered_df = filtered_df[filtered_df['discipline_title'] == discipline]
    if athlete != 'All':
        filtered_df = filtered_df[filtered_df['athlete_full_name'] == athlete]
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Athletes", len(filtered_df['athlete_full_name'].unique()))
    with col2:
        st.metric("Disciplines", len(filtered_df['discipline_title'].unique()))
    with col3:
        st.metric("Events", len(filtered_df['event_title'].unique()))
    
    # World Map Visualization
    st.subheader("Medal Distribution World Map")
    country_total_medals = filtered_df.groupby('country_name')['medal_type'].count().reset_index()
    country_total_medals.columns = ['country', 'total_medals']
    
    fig = px.choropleth(
        country_total_medals,
        locations='country',
        locationmode='country names',
        color='total_medals',
        hover_name='country',
        color_continuous_scale='Viridis',
        title=f'Total Medals by Country {f"({season})" if season != "All" else ""}'
    )
    
    fig.update_layout(
        height=600,
        title_x=0.5,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )
    st.plotly_chart(fig)
    
    # Total Medals by Country (Horizontal Stacked Bar)
    st.subheader("Total Medals by Country")
    country_medals = filtered_df.groupby('country_name')['medal_type'].value_counts().unstack(fill_value=0)
    country_medals['Total'] = country_medals.sum(axis=1)
    country_medals = country_medals.sort_values('Total', ascending=True).tail(30)
    
    fig = go.Figure()
    for medal_type in ['GOLD', 'SILVER', 'BRONZE']:
        if medal_type in country_medals.columns:
            fig.add_trace(go.Bar(
                y=country_medals.index,
                x=country_medals[medal_type],
                name=medal_type,
                orientation='h',
                text=country_medals[medal_type],
                textposition='auto',
                marker_color={'GOLD': '#FFD700', 'SILVER': '#C0C0C0', 'BRONZE': '#CD7F32'}[medal_type]
            ))
    
    annotations = []
    for i, row in enumerate(country_medals.itertuples()):
        annotations.append(dict(
            x=row.Total,
            y=row.Index,
            text=f'Total: {row.Total}',
            showarrow=False,
            xanchor='left',
            xshift=10
        ))
    
    fig.update_layout(
        barmode='stack',
        title=f'Medal Distribution by Country {f"({season})" if season != "All" else ""}',
        xaxis_title='Number of Medals',
        yaxis_title='Country',
        height=800,
        showlegend=True,
        yaxis={'categoryorder':'total ascending'},
        annotations=annotations
    )
    st.plotly_chart(fig)
    
    # Medal Count by Discipline
    st.subheader("Medal Count by Discipline")
    discipline_medals = filtered_df.groupby('discipline_title')['medal_type'].value_counts().unstack(fill_value=0)
    discipline_medals['Total'] = discipline_medals.sum(axis=1)
    discipline_medals = discipline_medals.sort_values('Total', ascending=False).head(20)
    
    fig = go.Figure()
    for medal_type in ['GOLD', 'SILVER', 'BRONZE']:
        if medal_type in discipline_medals.columns:
            fig.add_trace(go.Bar(
                x=discipline_medals.index,
                y=discipline_medals[medal_type],
                name=medal_type,
                text=discipline_medals[medal_type],
                textposition='auto',
                marker_color={'GOLD': '#FFD700', 'SILVER': '#C0C0C0', 'BRONZE': '#CD7F32'}[medal_type]
            ))
    
    annotations = []
    for i, (idx, row) in enumerate(discipline_medals.iterrows()):
        annotations.append(dict(
            x=idx,
            y=row['Total'],
            text=f'Total: {row["Total"]}',
            showarrow=False,
            yshift=10
        ))
    
    fig.update_layout(
        barmode='group',
        title=f'Top 20 Disciplines by Medal Count {f"({season})" if season != "All" else ""}',
        xaxis_title='Discipline',
        yaxis_title='Number of Medals',
        height=600,
        showlegend=True,
        xaxis_tickangle=45,
        annotations=annotations
    )
    st.plotly_chart(fig)
    
    # Gender and Medal Type Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gender Distribution")
        gender_counts = filtered_df['event_gender'].value_counts()
        fig = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title=f'Events by Gender {f"({season})" if season != "All" else ""}'
        )
        fig.update_traces(
            textposition='inside',
            textinfo='label+value+percent'
        )
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Medal Type Distribution")
        medal_counts = filtered_df['medal_type'].value_counts()
        fig = px.pie(
            values=medal_counts.values,
            names=medal_counts.index,
            title=f'Distribution of Medal Types {f"({season})" if season != "All" else ""}',
            color_discrete_map={'GOLD': '#FFD700', 'SILVER': '#C0C0C0', 'BRONZE': '#CD7F32'}
        )
        fig.update_traces(
            textposition='inside',
            textinfo='label+value+percent'
        )
        st.plotly_chart(fig)
    
    # Top Athletes
    st.subheader("Top Athletes by Medal Count")
    athlete_medals = filtered_df.groupby('athlete_full_name')['medal_type'].value_counts().unstack(fill_value=0)
    athlete_medals['Total'] = athlete_medals.sum(axis=1)
    athlete_medals = athlete_medals.sort_values('Total', ascending=True).tail(15)
    
    fig = go.Figure()
    for medal_type in ['GOLD', 'SILVER', 'BRONZE']:
        if medal_type in athlete_medals.columns:
            fig.add_trace(go.Bar(
                y=athlete_medals.index,
                x=athlete_medals[medal_type],
                name=medal_type,
                orientation='h',
                text=athlete_medals[medal_type],
                textposition='auto',
                marker_color={'GOLD': '#FFD700', 'SILVER': '#C0C0C0', 'BRONZE': '#CD7F32'}[medal_type]
            ))
    
    annotations = []
    for i, row in enumerate(athlete_medals.itertuples()):
        annotations.append(dict(
            x=row.Total,
            y=row.Index,
            text=f'Total: {row.Total}',
            showarrow=False,
            xanchor='left',
            xshift=10
        ))
    
    fig.update_layout(
        barmode='stack',
        title=f'Top 15 Athletes by Medal Count {f"({season})" if season != "All" else ""}',
        xaxis_title='Number of Medals',
        yaxis_title='Athlete',
        height=600,
        showlegend=True,
        annotations=annotations
    )
    st.plotly_chart(fig)