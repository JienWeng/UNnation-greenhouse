import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="UN Greenhouse Gas Emissions Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal custom CSS - keep default Streamlit styling
st.markdown("""
<style>
    /* Clean header */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all the processed datasets"""
    try:
        # Load main datasets
        master_df = pd.read_csv('greenhouse_gas_master_dataset.csv')
        long_df = pd.read_csv('greenhouse_gas_long_format.csv')
        country_stats = pd.read_csv('country_statistics.csv')
        dataset_summary = pd.read_csv('dataset_summary.csv')
        latest_emissions = pd.read_csv('latest_emissions_2021.csv')
        
        return master_df, long_df, country_stats, dataset_summary, latest_emissions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

def get_country_mapping():
    """Map country names to ISO codes for choropleth maps"""
    return {
        'United States of America': 'USA',
        'Russian Federation': 'RUS',
        'United Kingdom': 'GBR',
        'Republic of Korea': 'KOR',
        'Islamic Republic of Iran': 'IRN',
        'Türkiye': 'TUR',
        'Netherlands (Kingdom of the)': 'NLD'
    }

import plotly.express as px

def create_world_map(df, year, gas_type='total_ghg'):
    if gas_type not in df.columns:
        return None

    # Filter for the selected year
    year_data = df[df['Year'] == year].copy()

    # Map country names to ISO Alpha-3 codes
    country_mapping = get_country_mapping()
    year_data['iso_alpha'] = year_data['Country or Area'].map(country_mapping)

    # Attempt to fill missing codes with the first 3 letters as approximation
    mask = year_data['iso_alpha'].isna()
    year_data.loc[mask, 'iso_alpha'] = year_data.loc[mask, 'Country or Area'].str[:3].str.upper()

    # Create the choropleth map
    fig = px.choropleth(
        year_data,
        locations='iso_alpha',
        color=gas_type,
        hover_name='Country or Area',
        hover_data={gas_type: ':.0f'},
        color_continuous_scale='Viridis',
        title=f'{gas_type.upper().replace("_", " ")} Emissions by Country ({year})',
        labels={gas_type: 'Emissions (kt CO₂ eq)'}
    )

    fig.update_layout(
        title_x=0,
        title_font_size=20,
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='orthographic'
        )
    )

    return fig

def create_animated_timeline_chart(df, countries, gas_type='total_ghg'):
    """Create an animated timeline chart"""
    if gas_type not in df.columns or not countries:
        return None
    
    # Filter data for selected countries
    filtered_df = df[df['Country or Area'].isin(countries)].copy()
    
    # Create animated line plot
    fig = px.line(
        filtered_df,
        x='Year',
        y=gas_type,
        color='Country or Area',
        title=f'{gas_type.upper().replace("_", " ")} Emissions Timeline',
        labels={gas_type: 'Emissions (kt CO2 eq)', 'Country or Area': 'Country'},
        line_shape='spline'
    )
    
    fig.update_traces(
        mode='lines',
        line=dict(width=3)
    )
    
    fig.update_layout(
        title_x=0,
        title_font_size=18,
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_global_trends_chart(df):
    """Create clean global trends visualization"""
    gas_columns = ['co2', 'methane', 'hfc', 'total_ghg']
    available_cols = [col for col in gas_columns if col in df.columns]
    
    if not available_cols:
        return None
    
    # Calculate global totals by year
    global_totals = df.groupby('Year')[available_cols].sum().reset_index()
    
    # Create clean subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[col.upper().replace('_', ' ') for col in available_cols],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.3,
        horizontal_spacing=0.1
    )
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
    
    for i, gas in enumerate(available_cols):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(
            go.Scatter(
                x=global_totals['Year'],
                y=global_totals[gas],
                mode='lines',
                name=gas.upper(),
                line=dict(color=colors[i], width=2),
                hovertemplate=f'<b>{gas.upper()}</b><br>Year: %{{x}}<br>Emissions: %{{y:,.0f}} kt CO2 eq<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Global Greenhouse Gas Emissions Trends",
        title_x=0,
        title_font_size=18,
        height=500,
        showlegend=False
    )
    
    # Clean axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Year", row=i, col=j)
            fig.update_yaxes(title_text="Emissions (kt CO2 eq)", row=i, col=j)
    
    return fig

def create_top_emitters_chart(df, year, top_n=10):
    """Create clean top emitters chart"""
    year_data = df[df['Year'] == year]
    if 'total_ghg' not in df.columns:
        return None
    
    top_emitters = year_data.nlargest(top_n, 'total_ghg')
    
    # Shorten long country names
    top_emitters['Short_Country'] = top_emitters['Country or Area'].replace({
        'United States of America': 'USA',
        'Russian Federation': 'Russia',
        'United Kingdom': 'UK',
        'Islamic Republic of Iran': 'Iran',
        'Republic of Korea': 'South Korea',
        'Netherlands (Kingdom of the)': 'Netherlands'
    })
    
    fig = px.bar(
        top_emitters,
        x='total_ghg',
        y='Short_Country',
        orientation='h',
        title=f'Top {top_n} Emitters ({year})',
        labels={'total_ghg': 'Emissions (kt CO2 eq)', 'Short_Country': 'Country'}
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        title_x=0,
        title_font_size=16
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Emissions: %{x:,.0f} kt CO2 eq<extra></extra>'
    )
    
    return fig

def create_regional_comparison_chart(df):
    """Create modern regional comparison chart"""
    if 'Region' not in df.columns:
        return None
    
    gas_columns = ['co2', 'methane', 'hfc', 'total_ghg']
    available_cols = [col for col in gas_columns if col in df.columns]
    
    # Calculate average emissions by region
    regional_avg = df.groupby('Region')[available_cols].mean().reset_index()
    
    fig = go.Figure()
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
    
    for i, gas in enumerate(available_cols):
        fig.add_trace(go.Bar(
            name=gas.upper(),
            x=regional_avg['Region'],
            y=regional_avg[gas],
            marker_color=colors[i % len(colors)],
            marker_line_color='white',
            marker_line_width=1,
            hovertemplate=f'<b>%{{x}}</b><br>{gas.upper()}: %{{y:,.0f}} kt CO2 eq<extra></extra>'
        ))
    
    fig.update_layout(
        title='Regional Emissions Comparison',
        title_x=0,
        title_font_size=18,
        title_font_weight='bold',
        xaxis_title='Region',
        yaxis_title='Average Emissions (kt CO2 eq)',
        barmode='stack',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_time_series_chart(df, countries, gas_type='total_ghg'):
    """Create modern time series chart with animations"""
    if gas_type not in df.columns:
        return None
    
    fig = go.Figure()
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
    
    for i, country in enumerate(countries):
        country_data = df[df['Country or Area'] == country].sort_values('Year')
        
        # Shorten country names
        short_name = country.replace('United States of America', 'USA').replace('Russian Federation', 'Russia')
        
        fig.add_trace(go.Scatter(
            x=country_data['Year'],
            y=country_data[gas_type],
            mode='lines',
            name=short_name,
            line=dict(color=colors[i % len(colors)], width=3, shape='spline'),
            hovertemplate=f'<b>{short_name}</b><br>Year: %{{x}}<br>Emissions: %{{y:,.0f}} kt CO2 eq<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'{gas_type.upper().replace("_", " ")} Emissions Timeline',
        title_x=0,
        title_font_size=18,
        title_font_weight='bold',
        xaxis_title='Year',
        yaxis_title=f'{gas_type.upper()} Emissions (kt CO2 eq)',
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    return fig

def create_correlation_heatmap(df):
    """Create modern correlation heatmap"""
    gas_columns = ['co2', 'methane', 'hfc', 'total_ghg']
    available_cols = [col for col in gas_columns if col in df.columns]
    
    if len(available_cols) < 2:
        return None
    
    correlation_matrix = df[available_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='Gas Type Correlations'
    )
    
    fig.update_layout(
        title_x=0,
        title_font_size=18,
        title_font_weight='bold',
        height=400
    )
    
    return fig

def create_emission_category_chart(df):
    """Create modern emission category chart"""
    if 'Emission_Category' not in df.columns:
        return None
    
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    
    category_counts = latest_data['Emission_Category'].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title=f'Country Distribution by Emission Level ({latest_year})',
        color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
    )
    
    fig.update_layout(
        title_x=0,
        title_font_size=16,
        title_font_weight='bold',
        height=400
    )
    
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Countries: %{value}<br>Percentage: %{percent}<extra></extra>',
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig

def create_gas_composition_chart(df, countries, year):
    """Create gas composition chart for selected countries"""
    gas_columns = ['co2', 'methane', 'hfc']
    available_cols = [col for col in gas_columns if col in df.columns]
    
    if not available_cols or not countries:
        return None
    
    year_data = df[(df['Year'] == year) & (df['Country or Area'].isin(countries))]
    
    # Prepare data for stacked bar chart
    chart_data = []
    for country in countries:
        country_data = year_data[year_data['Country or Area'] == country]
        if not country_data.empty:
            for gas in available_cols:
                chart_data.append({
                    'Country': country.replace('United States of America', 'USA').replace('Russian Federation', 'Russia'),
                    'Gas': gas.upper(),
                    'Emissions': country_data[gas].iloc[0]
                })
    
    if not chart_data:
        return None
    
    chart_df = pd.DataFrame(chart_data)
    
    fig = px.bar(
        chart_df,
        x='Country',
        y='Emissions',
        color='Gas',
        title=f'Gas Composition by Country ({year})',
        labels={'Emissions': 'Emissions (kt CO2 eq)'},
        color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b']
    )
    
    fig.update_layout(
        title_x=0,
        title_font_size=18,
        title_font_weight='bold',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def main():
    # Load data
    master_df, long_df, country_stats, dataset_summary, latest_emissions = load_data()
    
    if master_df is None:
        st.error("Failed to load data. Please check if the CSV files exist.")
        return
    
    # Main header
    st.markdown('<h1 class="main-header">UN Greenhouse Gas Emissions Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for ALL controls
    st.sidebar.header("Dashboard Controls")
    
    # Dataset overview
    if dataset_summary is not None and not dataset_summary.empty:
        st.sidebar.markdown("### Dataset Overview")
        summary_data = dataset_summary.iloc[0]
        st.sidebar.info(f"""
        **Countries:** {summary_data['total_countries']}  
        **Years:** {summary_data['year_range']}  
        **Records:** {summary_data['total_records']:,}  
        **Gas Types:** {summary_data['gas_types']}
        """)
    
    # Interactive controls
    st.sidebar.markdown("### Interactive Controls")
    
    # Year slider for timeline
    years = sorted(master_df['Year'].unique())
    selected_year = st.sidebar.slider(
        "Select Year",
        min_value=int(years[0]),
        max_value=int(years[-1]),
        value=int(years[-1]),
        step=1
    )
    
    # Year range for timeline analysis
    year_range = st.sidebar.slider(
        "Year Range for Analysis",
        min_value=int(years[0]),
        max_value=int(years[-1]),
        value=(int(years[0]), int(years[-1])),
        step=1
    )
    
    # Country selection with better defaults
    countries = sorted(master_df['Country or Area'].unique())
    
    # Remove problematic default countries and add better ones
    default_countries = []
    preferred_countries = ['United States of America', 'Germany', 'Japan', 'Canada', 'Australia', 'United Kingdom']
    for country in preferred_countries:
        if country in countries:
            default_countries.append(country)
    
    selected_countries = st.sidebar.multiselect(
        "Select Countries for Comparison",
        countries,
        default=default_countries[:4]  # Limit to 4 for better visualization
    )
    
    # Gas type selection
    gas_columns = ['co2', 'methane', 'hfc', 'total_ghg']
    available_gas_cols = [col for col in gas_columns if col in master_df.columns]
    gas_labels = {'co2': 'CO2', 'methane': 'Methane', 'hfc': 'HFC', 'total_ghg': 'Total GHG'}
    
    selected_gas = st.sidebar.selectbox(
        "Gas Type for Analysis",
        available_gas_cols,
        format_func=lambda x: gas_labels.get(x, x) or x,
        index=len(available_gas_cols)-1 if available_gas_cols else 0
    )
    
    # Top N slider in sidebar
    top_n_countries = st.sidebar.slider("Top N Countries to Display", 5, 20, 10)
    
    # Filter data based on selections
    filtered_df = master_df[(master_df['Year'] >= year_range[0]) & (master_df['Year'] <= year_range[1])]
    
    # Key metrics
    if latest_emissions is not None and not latest_emissions.empty:
        col1, col2, col3 = st.columns(3)
        
        latest_year_data = latest_emissions
        
        with col1:
            if 'total_ghg' in latest_year_data.columns:
                total_global = latest_year_data['total_ghg'].sum()
                st.metric(
                    label="Global Total GHG (2021)",
                    value=f"{total_global/1000000:.1f}M",
                    help="Million kt CO2 equivalent"
                )
        
        with col2:
            if 'total_ghg' in latest_year_data.columns:
                top_emitter = latest_year_data.loc[latest_year_data['total_ghg'].idxmax(), 'Country or Area']
                top_emissions = latest_year_data['total_ghg'].max()
                # Shorten the country name
                display_name = str(top_emitter).replace('United States of America', 'USA').replace('Russian Federation', 'Russia')
                st.metric(
                    label="Top Emitter (2021)",
                    value=display_name,
                    delta=f"{top_emissions/1000:.0f}k kt"
                )
        
        with col3:
            if 'co2' in latest_year_data.columns and 'total_ghg' in latest_year_data.columns:
                co2_proportion = (latest_year_data['co2'].sum() / latest_year_data['total_ghg'].sum()) * 100
                st.metric(
                    label="CO2 Share of Total",
                    value=f"{co2_proportion:.1f}%",
                    help="CO2 percentage of total GHG emissions"
                )
    
    # Main content with 4 tabs only (removed Regional View and Advanced Analytics)
    tab1, tab2, tab3, tab4 = st.tabs([
        "Global Overview",
        "World Map", 
        "Top Emitters",
        "Country Analysis"
    ])
    
    with tab1:
        st.markdown("### Global Emission Trends")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            trends_chart = create_global_trends_chart(filtered_df)
            if trends_chart:
                st.plotly_chart(trends_chart, use_container_width=True)
        
        with col2:
            # Quick insights
            if 'total_ghg' in filtered_df.columns:
                global_totals = filtered_df.groupby('Year')['total_ghg'].sum()
                if len(global_totals) >= 2:
                    recent_change = ((global_totals.iloc[-1] - global_totals.iloc[-2]) / global_totals.iloc[-2]) * 100
                    
                    st.markdown("#### Key Insights")
                    st.info(f"""
                    **Year-over-Year Change**  
                    {recent_change:+.2f}% from {global_totals.index[-2]} to {global_totals.index[-1]}
                    """)
                    
                    if len(global_totals) >= 5:
                        five_year_change = ((global_totals.iloc[-1] - global_totals.iloc[-5]) / global_totals.iloc[-5]) * 100
                        st.success(f"""
                        **5-Year Trend**  
                        {five_year_change:+.2f}% change over 5 years
                        """)
    
    with tab2:
        st.markdown("### Global Emissions Map")
        
        # Use sidebar controls for map
        world_map = create_world_map(master_df, selected_year, selected_gas)
        if world_map:
            st.plotly_chart(world_map, use_container_width=True)
        else:
            st.warning("Unable to create world map. Please check the data.")
    
    with tab3:
        st.markdown("### Top Emitting Countries")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            top_emitters_chart = create_top_emitters_chart(master_df, selected_year, top_n_countries)
            if top_emitters_chart:
                st.plotly_chart(top_emitters_chart, use_container_width=True)
        
        with col2:
            # Emission categories
            category_chart = create_emission_category_chart(master_df)
            if category_chart:
                st.plotly_chart(category_chart, use_container_width=True)
        
        # Rankings table
        st.markdown("#### Detailed Rankings")
        year_data = master_df[master_df['Year'] == selected_year]
        if 'total_ghg' in year_data.columns:
            top_table = year_data.nlargest(15, 'total_ghg')[['Country or Area', 'total_ghg', 'co2', 'methane', 'hfc']].round(0)
            
            # Shorten country names in table
            top_table['Country or Area'] = top_table['Country or Area'].replace({
                'United States of America': 'USA',
                'Russian Federation': 'Russia',
                'United Kingdom': 'UK',
                'Islamic Republic of Iran': 'Iran'
            })
            
            top_table.columns = ['Country', 'Total GHG', 'CO2', 'Methane', 'HFC']
            st.dataframe(top_table, use_container_width=True, hide_index=True)
    
    with tab4:
        st.markdown("### Country Comparison & Timeline")
        
        if selected_countries:
            # Timeline chart
            timeline_chart = create_animated_timeline_chart(filtered_df, selected_countries, selected_gas)
            if timeline_chart:
                st.plotly_chart(timeline_chart, use_container_width=True)
            
            # Gas composition comparison
            col1, col2 = st.columns(2)
            
            with col1:
                composition_chart = create_gas_composition_chart(master_df, selected_countries, selected_year)
                if composition_chart:
                    st.plotly_chart(composition_chart, use_container_width=True)
            
            with col2:
                # Country comparison table
                st.markdown("#### Current Year Comparison")
                comparison_data = master_df[
                    (master_df['Country or Area'].isin(selected_countries)) & 
                    (master_df['Year'] == selected_year)
                ]
                
                if not comparison_data.empty:
                    display_cols = ['Country or Area'] + available_gas_cols
                    available_display_cols = [col for col in display_cols if col in comparison_data.columns]
                    comparison_table = comparison_data[available_display_cols].round(0)
                    
                    # Shorten country names
                    comparison_table['Country or Area'] = comparison_table['Country or Area'].replace({
                        'United States of America': 'USA',
                        'Russian Federation': 'Russia'
                    })
                    
                    st.dataframe(comparison_table, use_container_width=True, hide_index=True)
        else:
            st.info("Please select countries from the sidebar to see comparisons.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p><strong>UN Greenhouse Gas Emissions Dashboard</strong> | Data: United Nations Statistics Division</p>
        <p>Built by Jien Weng 2025</p>
        <p>
            <a href="https://www.linkedin.com/in/jienweng" target="_blank" style="margin: 0 10px;">LinkedIn</a> |
            <a href="https://jienweng.com" target="_blank" style="margin: 0 10px;">Website</a> |
            <a href="https://github.com/jienweng" target="_blank" style="margin: 0 10px;">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
