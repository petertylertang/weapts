import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Residences at West Edge Analytics",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏢 Residences at West Edge - Market Analytics Dashboard")
st.markdown("---")

@st.cache_data
def load_and_process_data():
    """Load and preprocess the apartment data"""
    # Construct the URL to download the CSV from Google Drive
    file_id = '1-S127zjAjfaqGT-cZzM5PGYfCTbH5HKJ'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    df = pd.read_csv(url)
    
    # Parse dates
    df['date_scraped'] = pd.to_datetime(df['date_scraped'])
    df['date'] = df['date_scraped'].dt.date
    
    # Clean price data
    df['price_numeric'] = pd.to_numeric(df['price'], errors='coerce')
    df['has_price'] = ~df['price_numeric'].isna()
    
    # Create apartment ID
    df['apartment_id'] = df['apartment_type'] + '-' + df['apartment_room'].astype(str)
    
    # Parse workforce housing
    df['is_workforce_housing'] = df['is_workforce_housing'] == 'True'
    
    # Calculate price per sqft
    df['price_per_sqft'] = df['price_numeric'] / df['sq_ft']
    
    # Create bedroom labels
    df['bedroom_label'] = df['num_bedrooms'].map({
        0: 'Studio',
        1: '1 Bedroom',
        2: '2 Bedrooms',
        3: '3 Bedrooms'
    }).fillna(df['num_bedrooms'].astype(str) + ' Bedrooms')
    
    return df

# Load data
df = load_and_process_data()

# Get the latest date for the footer
available_dates = sorted(df['date'].unique())
latest_date = available_dates[-1] if available_dates else datetime.now().date()

# Sidebar filters
st.sidebar.header("🔍 Filters")
st.sidebar.markdown("*Filters now apply to all tabs.*")

# The source_df is the entire dataset. Filters are applied on top of this.
source_df = df.copy()

date_range = st.sidebar.date_input(
    "Date Range for Analysis",
    value=[df['date'].min(), df['date'].max()],
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Get unique values from the full dataset for filters
all_bedroom_types = sorted(df['bedroom_label'].unique())
all_apartment_types = sorted(df['apartment_type'].unique())
all_floors = sorted(df['floor'].unique())

bedroom_filter = st.sidebar.multiselect(
    "Bedroom Type",
    options=all_bedroom_types,
    default=all_bedroom_types
)

apartment_type_filter = st.sidebar.multiselect(
    "Apartment Type",
    options=all_apartment_types,
    default=all_apartment_types
)

floor_filter = st.sidebar.multiselect(
    "Floor",
    options=all_floors,
    default=all_floors
)

# Price range filter
price_min = int(df['price_numeric'].dropna().min()) if df['price_numeric'].notna().any() else 0
price_max = int(df['price_numeric'].dropna().max()) if df['price_numeric'].notna().any() else 0
price_range = st.sidebar.slider(
    "Price Range",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    step=50,
)

# Apply filters for historical analysis
mask = (
    (source_df['date'] >= date_range[0]) &
    (source_df['date'] <= date_range[1]) &
    (source_df['bedroom_label'].isin(bedroom_filter)) &
    (source_df['apartment_type'].isin(apartment_type_filter)) &
    (source_df['floor'].isin(floor_filter)) &
    (
        (~source_df['has_price']) |
        ((source_df['price_numeric'] >= price_range[0]) &
         (source_df['price_numeric'] <= price_range[1]))
    )
)
filtered_df = source_df[mask].copy()

# Create a filtered snapshot for the overview tab based on the END of the date range
snapshot_date = date_range[1]
snapshot_df = source_df[(source_df['date'] == snapshot_date) &
                 (source_df['bedroom_label'].isin(bedroom_filter)) &
                 (source_df['apartment_type'].isin(apartment_type_filter)) &
                 (source_df['floor'].isin(floor_filter)) &
                 (
                     (~source_df['has_price']) |
                     ((source_df['price_numeric'] >= price_range[0]) &
                      (source_df['price_numeric'] <= price_range[1]))
                 )].copy()

# Main dashboard tabs
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["☀️ Daily Briefing", "📊 Overview", "💰 Price Trends", "🏠 Unit Analysis", "⏱️ Market Duration", "🎯 Insights", "🤖 AI Recommendations"])

# Tab 0: Daily Briefing
with tab0:
    st.subheader("Market Activity Briefing")
    st.caption("*Briefing is based on the filters selected in the sidebar.*")
    briefing_period = st.radio(
        "Select Briefing Period",
        ('Daily', 'Weekly', 'Monthly'),
        horizontal=True,
        key='briefing_period'
    )

    # --- Briefing Data Prep ---
    available_dates_pd = pd.to_datetime(pd.Series(df['date'].unique())).sort_values(ascending=False)
    latest_date_pd = available_dates_pd.iloc[0] if not available_dates_pd.empty else None
    past_dates = available_dates_pd[1:]
    previous_date_pd = None

    if latest_date_pd is not None:
        if briefing_period == 'Daily':
            if not past_dates.empty:
                previous_date_pd = past_dates.iloc[0]
        
        elif briefing_period == 'Weekly':
            target_date = latest_date_pd - timedelta(days=7)
            if not past_dates.empty:
                # Find the one closest to the target date
                previous_date_pd = past_dates.iloc[(past_dates - target_date).abs().argmin()]

        elif briefing_period == 'Monthly':
            target_date = latest_date_pd - timedelta(days=30)
            if not past_dates.empty:
                previous_date_pd = past_dates.iloc[(past_dates - target_date).abs().argmin()]

    # Convert back to date objects for consistency
    briefing_latest_date = latest_date_pd.date() if latest_date_pd is not None else datetime.now().date()
    briefing_previous_date = previous_date_pd.date() if pd.notnull(previous_date_pd) else None

    if briefing_previous_date:
        st.caption(f"Comparing data from **{briefing_previous_date.strftime('%B %d, %Y')}** and **{briefing_latest_date.strftime('%B %d, %Y')}**")

        # Get the data for the two dates from the full source_df
        latest_data = source_df[source_df['date'] == briefing_latest_date].copy()
        previous_data = source_df[source_df['date'] == briefing_previous_date].copy()

        # Apply the sidebar filters to both dataframes
        briefing_latest_df = latest_data[
            (latest_data['bedroom_label'].isin(bedroom_filter)) &
            (latest_data['apartment_type'].isin(apartment_type_filter)) &
            (latest_data['floor'].isin(floor_filter))
        ].copy()

        briefing_previous_df = previous_data[
            (previous_data['bedroom_label'].isin(bedroom_filter)) &
            (previous_data['apartment_type'].isin(apartment_type_filter)) &
            (previous_data['floor'].isin(floor_filter))
        ].copy()

        # Calculate changes
        latest_units_set = set(briefing_latest_df['apartment_id'].unique())
        previous_units_set = set(briefing_previous_df['apartment_id'].unique())

        new_units_list = list(latest_units_set - previous_units_set)
        exited_units_list = list(previous_units_set - latest_units_set)

        common_units = latest_units_set.intersection(previous_units_set)
        
        latest_prices = briefing_latest_df[briefing_latest_df['apartment_id'].isin(common_units)][['apartment_id', 'price_numeric']].drop_duplicates(subset=['apartment_id'])
        previous_prices = briefing_previous_df[briefing_previous_df['apartment_id'].isin(common_units)][['apartment_id', 'price_numeric']].drop_duplicates(subset=['apartment_id'])
        
        price_comp_df = pd.merge(latest_prices, previous_prices, on='apartment_id', suffixes=('_new', '_old'))
        price_comp_df = price_comp_df.dropna(subset=['price_numeric_new', 'price_numeric_old'])
        price_comp_df['price_diff'] = price_comp_df['price_numeric_new'] - price_comp_df['price_numeric_old']
        
        price_increases_df = price_comp_df[price_comp_df['price_diff'] > 0].copy()
        price_decreases_df = price_comp_df[price_comp_df['price_diff'] < 0].copy()

        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"New Listings ({briefing_period})", len(new_units_list))
        col2.metric(f"Leased/Exited ({briefing_period})", len(exited_units_list))
        col3.metric(f"Price Increases ({briefing_period})", len(price_increases_df))
        col4.metric(f"Price Decreases ({briefing_period})", len(price_decreases_df))
        
        st.markdown("---")

        # Detailed Changes
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 Availability Changes")
            st.markdown("##### New Listings")
            if new_units_list:
                new_units_df = briefing_latest_df[briefing_latest_df['apartment_id'].isin(new_units_list)][['apartment_id', 'apartment_type', 'floor', 'bedroom_label', 'price_numeric']].drop_duplicates(subset=['apartment_id'])
                new_units_df.rename(columns={'apartment_id': 'Unit', 'apartment_type': 'Type', 'floor': 'Floor', 'bedroom_label': 'Bedrooms', 'price_numeric': 'Price'}, inplace=True)
                st.dataframe(new_units_df, hide_index=True, use_container_width=True)
            else:
                st.info(f"No new listings in this period.")

            st.markdown("##### Leased / Exited Listings")
            if exited_units_list:
                exited_units_df = briefing_previous_df[briefing_previous_df['apartment_id'].isin(exited_units_list)][['apartment_id', 'apartment_type', 'floor', 'bedroom_label']].drop_duplicates(subset=['apartment_id'])
                exited_units_df.rename(columns={'apartment_id': 'Unit', 'apartment_type': 'Type', 'floor': 'Floor', 'bedroom_label': 'Bedrooms'}, inplace=True)
                st.dataframe(exited_units_df, hide_index=True, use_container_width=True)
            else:
                st.info(f"No units leased or taken off market in this period.")
        with col2:
            st.markdown("#### 💰 Price Changes")
            st.markdown("##### Price Increases")
            if not price_increases_df.empty:
                inc_display = price_increases_df[['apartment_id', 'price_numeric_old', 'price_numeric_new', 'price_diff']].copy()
                inc_display.rename(columns={'apartment_id': 'Unit', 'price_numeric_old': 'Old Price', 'price_numeric_new': 'New Price', 'price_diff': 'Change'}, inplace=True)
                st.dataframe(inc_display.sort_values('Change', ascending=False), hide_index=True, use_container_width=True)
            else:
                st.info(f"No price increases in this period.")

            st.markdown("##### Price Decreases")
            if not price_decreases_df.empty:
                dec_display = price_decreases_df[['apartment_id', 'price_numeric_old', 'price_numeric_new', 'price_diff']].copy()
                dec_display.rename(columns={'apartment_id': 'Unit', 'price_numeric_old': 'Old Price', 'price_numeric_new': 'New Price', 'price_diff': 'Change'}, inplace=True)
                st.dataframe(dec_display.sort_values('Change', ascending=True), hide_index=True, use_container_width=True)
            else:
                st.info(f"No price decreases in this period.")
    else:
        st.info(f"Not enough historical data for a {briefing_period.lower()} briefing. At least two days of data are required.")

# Tab 1: Overview
with tab1:
    st.markdown("### Market Snapshot")
    st.caption(f"*Snapshot as of {snapshot_date.strftime('%B %d, %Y')} based on active filters*")
    
    # Calculate metrics from the filtered snapshot_df
    snapshot_units = snapshot_df['apartment_id'].nunique()
    snapshot_with_price = snapshot_df[snapshot_df['has_price']]
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Available Units", f"{snapshot_units}")
    with col2:
        if len(snapshot_with_price) > 0:
            avg_price = snapshot_with_price.groupby('apartment_id')['price_numeric'].first().mean()
            st.metric("Average Price", f"${avg_price:,.0f}")
        else:
            st.metric("Average Price", "N/A")
    with col3:
        # Calculate avg days on market for units in the snapshot
        snapshot_market_days = []
        for apt_id in snapshot_df['apartment_id'].unique():
            # Use the full df to get the complete history of the unit
            apt_history = source_df[source_df['apartment_id'] == apt_id]
            # Days on market up to the snapshot date
            days = (snapshot_date - apt_history['date'].min()).days + 1
            snapshot_market_days.append(days)
        avg_days = np.mean(snapshot_market_days) if snapshot_market_days else 0
        st.metric("Avg Days on Market", f"{avg_days:.0f}")
    with col4:
        if len(snapshot_with_price) > 0:
            avg_psf = snapshot_with_price.groupby('apartment_id')['price_per_sqft'].first().mean()
            st.metric("Avg Price/SqFt", f"${avg_psf:.2f}")
        else:
            st.metric("Avg Price/SqFt", "N/A")
    
    st.markdown("---")
    
    # Market activity over time
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📈 Market Activity Timeline (Filtered)")
        if not filtered_df.empty:
            daily_counts = filtered_df.groupby('date')['apartment_id'].nunique().reset_index(name='count')
            fig = px.line(daily_counts, x='date', y='count', 
                         title="Daily Active Unique Listings",
                         labels={'count': 'Number of Units', 'date': 'Date'})
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity to display for the selected filters and date range.")
    
    with col2:
        st.subheader("🏢 Inventory by Type")
        st.caption(f"Snapshot on {snapshot_date.strftime('%b %d, %Y')}")
        if not snapshot_df.empty:
            type_counts = snapshot_df.groupby('apartment_type')['apartment_id'].nunique().reset_index(name='count')
            type_counts = type_counts.sort_values('count', ascending=True)
            
            fig = px.bar(type_counts, 
                        x='count', y='apartment_type', orientation='h',
                        title=f"Units by Type ({snapshot_units} total)",
                        labels={'count': 'Number of Units', 'apartment_type': 'Type'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No units available for this snapshot.")
    
    # Bedroom distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛏️ Bedroom Distribution (Snapshot)")
        if not snapshot_df.empty:
            bedroom_counts = snapshot_df.groupby('bedroom_label')['apartment_id'].nunique().reset_index(name='count')
            
            fig = px.pie(bedroom_counts, values='count', names='bedroom_label',
                        title=f"Inventory by Bedrooms ({snapshot_units} total)")
            fig.update_traces(textposition='inside', textinfo='value+percent')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No units available for this snapshot.")
    
    with col2:
        st.subheader("🏗️ Floor Distribution (Snapshot)")
        if not snapshot_df.empty:
            floor_counts = snapshot_df.groupby('floor')['apartment_id'].nunique().reset_index(name='count')
            floor_counts = floor_counts.sort_values('floor')
            
            fig = px.bar(floor_counts, x='floor', y='count',
                        title=f"Units by Floor ({snapshot_units} total)",
                        labels={'count': 'Number of Units', 'floor': 'Floor'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No units available for this snapshot.")
    
    # Show actual current units list
    st.markdown("---")
    st.subheader("📋 Unit List")

    view_mode = st.radio(
        "Select Unit View",
        ["Snapshot (Units available on selected end date)", "Historical (All units active during date range)"],
        horizontal=True,
        key="overview_view_mode",
        label_visibility="collapsed"
    )

    if "Snapshot" in view_mode:
        st.caption(f"Showing units available on {snapshot_date.strftime('%B %d, %Y')} based on active filters.")
        if not snapshot_df.empty:
            snapshot_summary = []
            for apt_id in snapshot_df['apartment_id'].unique():
                apt_data = snapshot_df[snapshot_df['apartment_id'] == apt_id].iloc[0]
                snapshot_summary.append({
                    'Unit': apt_id,
                    'Type': apt_data['apartment_type'],
                    'Floor': apt_data['floor'],
                    'Bedrooms': apt_data['bedroom_label'],
                    'Sq Ft': apt_data['sq_ft'],
                    'Price': f"${apt_data['price_numeric']:,.0f}" if apt_data['has_price'] else "Call for pricing"
                })
            
            snapshot_summary_df = pd.DataFrame(snapshot_summary)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                show_all = st.checkbox("Show all units", value=False, key="overview_show_all")
            
            if show_all:
                st.dataframe(snapshot_summary_df.sort_values(['Type', 'Floor']), use_container_width=True, hide_index=True)
            else:
                st.dataframe(snapshot_summary_df.sort_values(['Type', 'Floor']).head(10), use_container_width=True, hide_index=True)
                st.caption(f"Showing 10 of {len(snapshot_summary_df)} units. Check 'Show all units' to see complete list.")
        else:
            st.info("No units to display for the selected filters and snapshot date.")
    else: # Historical view
        st.caption(f"Showing all units active between {date_range[0].strftime('%b %d, %Y')} and {date_range[1].strftime('%b %d, %Y')}.")
        if not filtered_df.empty:
            historical_summary_source = filtered_df.sort_values('date').groupby('apartment_id').tail(1)
            snapshot_unit_ids = set(snapshot_df['apartment_id'].unique())
            
            historical_summary = []
            for index, apt_data in historical_summary_source.iterrows():
                apt_id = apt_data['apartment_id']
                historical_summary.append({
                    'Unit': apt_id,
                    'Status': '✅ Available' if apt_id in snapshot_unit_ids else '❌ Exited',
                    'Type': apt_data['apartment_type'],
                    'Floor': apt_data['floor'],
                    'Bedrooms': apt_data['bedroom_label'],
                    'Sq Ft': apt_data['sq_ft'],
                    'Last Known Price': f"${apt_data['price_numeric']:,.0f}" if apt_data['has_price'] else "Call for pricing"
                })
            
            historical_summary_df = pd.DataFrame(historical_summary)
            display_df = historical_summary_df.sort_values(['Status', 'Type', 'Floor'], ascending=[False, True, True])
            
            col1, col2 = st.columns([1, 3])
            with col1:
                show_all = st.checkbox("Show all units", value=False, key="overview_show_all_historical")
            
            if show_all:
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)
                st.caption(f"Showing 10 of {len(display_df)} units. Check 'Show all units' to see complete list.")
        else:
            st.info("No units to display for the selected filters and date range.")

# Tab 2: Price Trends
with tab2:
    st.markdown("### Price Analysis")
    st.caption("*Based on filtered date range and unit types*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Overall Price Trends")
        if len(filtered_df[filtered_df['has_price']]) > 0:
            daily_prices = filtered_df[filtered_df['has_price']].groupby('date').agg({
                'price_numeric': ['mean', 'median', 'min', 'max']
            }).reset_index()
            daily_prices.columns = ['date', 'mean', 'median', 'min', 'max']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_prices['date'], y=daily_prices['mean'],
                                    name='Average', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=daily_prices['date'], y=daily_prices['median'],
                                    name='Median', line=dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=daily_prices['date'], y=daily_prices['max'],
                                    name='Max', line=dict(color='red', dash='dot')))
            fig.add_trace(go.Scatter(x=daily_prices['date'], y=daily_prices['min'],
                                    name='Min', line=dict(color='orange', dash='dot')))
            fig.update_layout(title="Price Trends Over Time",
                             xaxis_title="Date", yaxis_title="Price ($)",
                             hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available for selected filters")
    
    with col2:
        st.subheader("💵 Price by Bedroom Count")
        if len(filtered_df[filtered_df['has_price']]) > 0:
            bedroom_prices = filtered_df[filtered_df['has_price']].groupby(['date', 'bedroom_label']).agg({
                'price_numeric': 'mean'
            }).reset_index()
            
            fig = px.line(bedroom_prices, x='date', y='price_numeric', color='bedroom_label',
                         title="Average Price by Bedroom Count",
                         labels={'price_numeric': 'Average Price ($)', 'bedroom_label': 'Type'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available for selected filters")
    
    # Price analysis for current inventory
    st.markdown("---")
    st.subheader(f"💰 Price Analysis Snapshot ({snapshot_date.strftime('%b %d, %Y')})")
    st.caption("*Based on all active filters*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price by apartment type (current)
        if not snapshot_df.empty and snapshot_df['has_price'].any():
            price_by_type = snapshot_df[snapshot_df['has_price']].groupby('apartment_type').agg({
                'price_numeric': ['mean', 'count']
            }).reset_index()
            price_by_type.columns = ['Type', 'Avg Price', 'Units']
            price_by_type = price_by_type.sort_values('Avg Price', ascending=False)
            
            fig = px.bar(price_by_type.head(10), x='Type', y='Avg Price',
                        title="Average Price by Type (Top 10)",
                        labels={'Avg Price': 'Average Price ($)'},
                        text='Units')
            fig.update_traces(texttemplate='%{text} units', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available for this snapshot.")
    
    with col2:
        # Price distribution (current)
        if not snapshot_df.empty and snapshot_df['has_price'].any():
            fig = px.histogram(snapshot_df[snapshot_df['has_price']], x='price_numeric', 
                              nbins=15,
                              title="Price Distribution",
                              labels={'price_numeric': 'Price ($)', 'count': 'Number of Units'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available for this snapshot.")
    
    # Price per square foot analysis
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 $/SqFt by Bedroom Type")
        if not snapshot_df.empty and snapshot_df['has_price'].any():
            psf_by_bedroom = snapshot_df[snapshot_df['has_price']].groupby('bedroom_label').agg({
                'price_per_sqft': 'mean',
                'apartment_id': 'nunique'
            }).reset_index()
            psf_by_bedroom.columns = ['Bedroom Type', 'Avg $/SqFt', 'Units']
            
            fig = px.bar(psf_by_bedroom, x='Bedroom Type', y='Avg $/SqFt',
                        title="Price per SqFt by Bedroom Type",
                        text='Units')
            fig.update_traces(texttemplate='%{text} units', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available")
    
    with col2:
        st.subheader("🏢 Price Range by Floor")
        if not snapshot_df.empty and snapshot_df['has_price'].any():
            floor_prices = snapshot_df[snapshot_df['has_price']].groupby('floor').agg({
                'price_numeric': ['min', 'max', 'mean']
            }).reset_index()
            floor_prices.columns = ['Floor', 'Min', 'Max', 'Avg']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=floor_prices['Floor'], y=floor_prices['Max'] - floor_prices['Min'],
                                base=floor_prices['Min'],
                                name='Price Range',
                                marker_color='lightblue'))
            fig.add_trace(go.Scatter(x=floor_prices['Floor'], y=floor_prices['Avg'],
                                    mode='markers',
                                    name='Average',
                                    marker=dict(size=10, color='red')))
            fig.update_layout(title="Price Range by Floor",
                            xaxis_title="Floor",
                            yaxis_title="Price ($)",
                            showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available")

# Tab 3: Unit Analysis
with tab3:
    st.subheader("🔍 Individual Unit Tracking (Filtered)")
    
    # Get units from the filtered dataset for analysis
    snapshot_unit_ids = set(snapshot_df['apartment_id'].unique())
    unit_ids_to_analyze = filtered_df['apartment_id'].unique()

    if len(unit_ids_to_analyze) > 0:
        all_units_stats = []
        for apt_id in unit_ids_to_analyze:
            apt_df = source_df[source_df['apartment_id'] == apt_id]
            apt_df_sorted = apt_df.sort_values('date')
            
            is_available = apt_id in snapshot_unit_ids
            
            prices = apt_df_sorted[apt_df_sorted['has_price']]['price_numeric'].values
            price_changes = 0
            if len(prices) > 1:
                price_changes = len(np.where(np.diff(prices) != 0)[0])
            
            all_units_stats.append({
                'Apartment ID': apt_id,
                'Type': apt_df_sorted.iloc[0]['apartment_type'],
                'Floor': apt_df_sorted.iloc[0]['floor'],
                'Bedroom Type': apt_df_sorted.iloc[0]['bedroom_label'],
                'Sq Ft': apt_df_sorted.iloc[0]['sq_ft'],
                'Available on Snapshot': '✅' if is_available else '❌',
                'First Seen': apt_df_sorted['date'].min(),
                'Last Seen': apt_df_sorted['date'].max(),
                'Days Tracked': (apt_df_sorted['date'].max() - apt_df_sorted['date'].min()).days + 1,
                'Initial Price': prices[0] if len(prices) > 0 else None,
                'Latest Price': prices[-1] if len(prices) > 0 else None,
                'Price Changes': price_changes,
                'Min Price': prices.min() if len(prices) > 0 else None,
                'Max Price': prices.max() if len(prices) > 0 else None
            })
        
        all_units_df = pd.DataFrame(all_units_stats)
        
        # Filter options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_available_only = st.checkbox("Show only available on snapshot date", value=False)
        with col2:
            sort_by = st.selectbox("Sort by", 
                                  ['Days Tracked', 'Price Changes', 'Latest Price', 'Apartment ID', 'Available on Snapshot'],
                                  key='unit_sort')
        with col3:
            sort_order = st.radio("Order", ['Descending', 'Ascending'], horizontal=True, key='unit_order')
        
        # Apply filters
        display_df = all_units_df.copy()
        if show_available_only:
            display_df = display_df[display_df['Available on Snapshot'] == '✅']
        
        # Sort
        ascending = sort_order == 'Ascending'
        display_df = display_df.sort_values(sort_by, ascending=ascending, na_position='last')
        
        # Display
        st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)
        st.caption(f"Showing top 20 of {len(display_df)} units from filtered results")
    else:
        st.info("No units match the current filter criteria.")
    
    # Unit selector for detailed view
    st.markdown("---")
    st.subheader("📈 Individual Unit Price History")
    
    # Get list of units from filtered results for selection
    all_unit_ids = sorted(filtered_df['apartment_id'].unique())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_unit = st.selectbox("Select Unit (from filtered results)", 
                                     all_unit_ids,
                                     key='unit_selector')
    with col2:
        if selected_unit:
            unit_status = "✅ Available on Snapshot" if selected_unit in snapshot_unit_ids else "❌ Exited"
            st.markdown(f"<br><h4>{unit_status}</h4>", unsafe_allow_html=True)
    
    if selected_unit:
        unit_df = df[df['apartment_id'] == selected_unit].sort_values('date')
        unit_df_prices = unit_df[unit_df['has_price']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price history chart
            if len(unit_df_prices) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=unit_df_prices['date'], y=unit_df_prices['price_numeric'],
                                        mode='lines+markers',
                                        name='Price',
                                        line=dict(color='blue', width=2),
                                        marker=dict(size=8)))
                
                # Add annotations for price changes
                if len(unit_df_prices) > 1:
                    prices = unit_df_prices['price_numeric'].values
                    dates = unit_df_prices['date'].values
                    for i in range(1, len(prices)):
                        if prices[i] != prices[i-1]:
                            change = prices[i] - prices[i-1]
                            change_pct = (change / prices[i-1]) * 100
                            color = 'green' if change > 0 else 'red'
                            fig.add_annotation(x=dates[i], y=prices[i],
                                             text=f"${change:+.0f}<br>({change_pct:+.1f}%)",
                                             showarrow=True,
                                             arrowhead=2,
                                             arrowcolor=color,
                                             font=dict(size=10, color=color))
                
                fig.update_layout(title=f"Price History for {selected_unit}",
                                xaxis_title="Date", yaxis_title="Price ($)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No price history available for this unit")
        
        with col2:
            # Unit details
            st.markdown("### Unit Details")
            unit_info = unit_df.iloc[0]
            st.markdown(f"""
            - **Type:** {unit_info['apartment_type']}
            - **Floor:** {unit_info['floor']}
            - **Room:** {unit_info['apartment_room']}
            - **Bedroom Type:** {unit_info['bedroom_label']}
            - **Bathrooms:** {unit_info['num_bathrooms']}
            - **Square Feet:** {unit_info['sq_ft']:,}
            - **Workforce Housing:** {'Yes' if unit_info['is_workforce_housing'] else 'No'}
            """)
            
            if len(unit_df_prices) > 0:
                st.markdown("### Price Statistics")
                st.markdown(f"""
                - **Latest Price:** ${unit_df_prices.iloc[-1]['price_numeric']:,.0f}
                - **Initial Price:** ${unit_df_prices.iloc[0]['price_numeric']:,.0f}
                - **Min Price:** ${unit_df_prices['price_numeric'].min():,.0f}
                - **Max Price:** ${unit_df_prices['price_numeric'].max():,.0f}
                - **Price Changes:** {len(np.where(np.diff(unit_df_prices['price_numeric'].values) != 0)[0])}
                - **Days Tracked:** {(unit_df['date'].max() - unit_df['date'].min()).days + 1}
                """)

# Tab 4: Market Duration
with tab4:
    st.subheader("⏱️ Time on Market Analysis (Filtered)")
    
    # Calculate market duration for each unit in the filtered dataset
    market_durations = []
    unit_ids_to_analyze = filtered_df['apartment_id'].unique()
    snapshot_unit_ids = set(snapshot_df['apartment_id'].unique())

    for apt_id in unit_ids_to_analyze:
        apt_df = source_df[source_df['apartment_id'] == apt_id]
        first_seen = apt_df['date'].min()
        
        # Check if unit is active on the snapshot date
        is_active = apt_id in snapshot_unit_ids
        
        if is_active:
            days_on_market = (snapshot_date - first_seen).days + 1
        else:
            # For exited units, it's their total lifetime
            days_on_market = (apt_df['date'].max() - first_seen).days + 1
        
        # Get first price
        apt_prices = apt_df[apt_df['has_price']].sort_values('date')
        initial_price = apt_prices.iloc[0]['price_numeric'] if len(apt_prices) > 0 else None
        
        market_durations.append({
            'apartment_id': apt_id,
            'bedroom_label': apt_df.iloc[0]['bedroom_label'],
            'days_on_market': days_on_market,
            'is_active': is_active,
        })
    
    duration_df = pd.DataFrame(market_durations)
    
    # Show summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    if not duration_df.empty:
        active_units = duration_df[duration_df['is_active']]
        exited_units = duration_df[~duration_df['is_active']]
        
        with col1:
            st.metric("Active Units (in snapshot)", len(active_units))
        with col2:
            st.metric("Exited Units (in range)", len(exited_units))
        with col3:
            if len(active_units) > 0:
                st.metric("Avg Days (Active)", f"{active_units['days_on_market'].mean():.0f}")
            else:
                st.metric("Avg Days (Active)", "N/A")
        with col4:
            if len(exited_units) > 0:
                st.metric("Avg Days (Exited)", f"{exited_units['days_on_market'].mean():.0f}")
            else:
                st.metric("Avg Days (Exited)", "N/A")
    else:
        st.info("No units to analyze for the selected filters.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    if not duration_df.empty:
        with col1:
            # Distribution of days on market
            fig = px.histogram(duration_df, x='days_on_market', 
                              color='is_active',
                              nbins=20,
                              title="Distribution of Days on Market",
                              labels={'days_on_market': 'Days on Market', 'count': 'Number of Units'},
                              color_discrete_map={True: 'blue', False: 'red'})
            fig.update_layout(legend_title_text='Status',
                             legend=dict(itemsizing='constant'))
            fig.for_each_trace(lambda t: t.update(name='Active' if t.name == 'True' else 'Exited'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Days on market by bedroom count
            fig = px.box(duration_df, x='bedroom_label', y='days_on_market',
                        color='is_active',
                        title="Days on Market by Bedroom Type",
                        labels={'days_on_market': 'Days on Market', 'bedroom_label': 'Bedroom Type'},
                        color_discrete_map={True: 'blue', False: 'red'})
            fig.update_layout(legend_title_text='Status')
            fig.for_each_trace(lambda t: t.update(name='Active' if t.name == 'True' else 'Exited'))
            st.plotly_chart(fig, use_container_width=True)
    
    # Survival curve
    st.markdown("---")
    st.subheader("📉 Market Survival Analysis (Filtered)")
    
    if not duration_df.empty:
        # Create survival data
        max_days = int(duration_df['days_on_market'].max())
        survival_data = []
        
        total_units_in_analysis = len(duration_df)
        for day in range(0, max_days + 1):
            # Count units still on market after 'day' days
            still_on_market = len(duration_df[duration_df['days_on_market'] >= day])
            survival_rate = still_on_market / total_units_in_analysis
            survival_data.append({'Day': day, 'Survival Rate': survival_rate})
        
        survival_df = pd.DataFrame(survival_data)
        
        fig = px.line(survival_df, x='Day', y='Survival Rate',
                     title="Probability of Remaining on Market Over Time",
                     labels={'Day': 'Days Since Listing', 'Survival Rate': 'Proportion Still on Market'})
        fig.update_layout(yaxis_tickformat='.0%')
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                     annotation_text="50% leased")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for survival analysis with current filters.")

# Tab 5: Insights
with tab5:
    st.subheader("🎯 Market Insights & Recommendations (Filtered)")
    st.caption(f"Insights are based on the selected filters and date range: {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')}")
    
    # Use filtered and snapshot data
    snapshot_units_count = snapshot_df['apartment_id'].nunique()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Snapshot Overview")
        snapshot_with_price = snapshot_df[snapshot_df['has_price']]
        units_with_price_count = snapshot_with_price['apartment_id'].nunique()
        
        if not snapshot_with_price.empty:
            # Calculate per-unit averages
            unit_prices = snapshot_with_price.groupby('apartment_id')['price_numeric'].first()
            avg_price = unit_prices.mean()
            min_price = unit_prices.min()
            max_price = unit_prices.max()
            
            st.markdown(f"""
            - **Inventory in Snapshot:** {snapshot_units_count} units
            - **Units with Price:** {units_with_price_count} ({units_with_price_count/snapshot_units_count*100:.1f}% if snapshot_units_count > 0 else 0)
            - **Average Asking Price:** ${avg_price:,.0f}
            - **Price Range:** ${min_price:,.0f} - ${max_price:,.0f}
            """)
        else:
            st.markdown(f"""
            - **Inventory in Snapshot:** {snapshot_units_count} units
            - **Units with Price:** 0
            - **Average Asking Price:** N/A
            - **Price Range:** N/A
            """)
    
    with col2:
        st.markdown("### 🔄 Turnover Metrics")
        total_units_in_range = filtered_df['apartment_id'].nunique()
        exited_units_count = total_units_in_range - snapshot_units_count
        
        # Calculate average days to lease for exited units
        exited_duration = []
        snapshot_unit_ids = set(snapshot_df['apartment_id'].unique())
        for apt_id in filtered_df['apartment_id'].unique():
            if apt_id not in snapshot_unit_ids:
                apt_df = source_df[source_df['apartment_id'] == apt_id]
                days = (apt_df['date'].max() - apt_df['date'].min()).days + 1
                exited_duration.append(days)
        
        avg_days_to_lease = np.mean(exited_duration) if exited_duration else 0
        
        st.markdown(f"""
        - **Total Units in Range:** {total_units_in_range}
        - **Units Leased/Exited:** {exited_units_count}
        - **Turnover Rate:** {exited_units_count/total_units_in_range*100:.1f}% if total_units_in_range > 0 else 0
        - **Avg Days to Lease:** {avg_days_to_lease:.0f} days
        """)
    
    with col3:
        st.markdown("### 💡 Price Dynamics")
        # Count units with price changes within the filtered range
        units_with_changes = 0
        total_reductions = 0
        total_increases = 0
        
        for apt_id in filtered_df['apartment_id'].unique():
            apt_df = filtered_df[(filtered_df['apartment_id'] == apt_id) & filtered_df['has_price']].sort_values('date')
            if len(apt_df) > 1:
                prices = apt_df['price_numeric'].values
                changes = np.diff(prices)
                non_zero_changes = changes[changes != 0]
                if len(non_zero_changes) > 0:
                    units_with_changes += 1
                    total_reductions += len(non_zero_changes[non_zero_changes < 0])
                    total_increases += len(non_zero_changes[non_zero_changes > 0])
        
        st.markdown(f"""
        - **Units with Price Changes:** {units_with_changes}
        - **Total Price Reductions:** {total_reductions}
        - **Total Price Increases:** {total_increases}
        - **Reduction/Increase Ratio:** {total_reductions/max(total_increases, 1):.2f}
        """)
    
    st.markdown("---")
    
    # Top movers
    st.subheader("🔥 Units with Most Price Activity (in Snapshot)")
    
    price_activity = []
    for apt_id in snapshot_df['apartment_id'].unique():
        apt_df = filtered_df[(filtered_df['apartment_id'] == apt_id) & filtered_df['has_price']].sort_values('date')
        if len(apt_df) > 1:
            prices = apt_df['price_numeric'].values
            changes = np.diff(prices)
            non_zero_changes = changes[changes != 0]
            
            if len(non_zero_changes) > 0:
                total_change = prices[-1] - prices[0]
                pct_change = (total_change / prices[0]) * 100
                
                price_activity.append({
                    'Unit': apt_id,
                    'Type': apt_df.iloc[0]['apartment_type'],
                    'Floor': apt_df.iloc[0]['floor'],
                    'Changes': len(non_zero_changes),
                    'Initial': f"${prices[0]:,.0f}",
                    'Current': f"${prices[-1]:,.0f}",
                    'Total Change': f"${total_change:+,.0f}",
                    '% Change': f"{pct_change:+.1f}%"
                })
    
    if price_activity:
        activity_df = pd.DataFrame(price_activity)
        activity_df = activity_df.sort_values('Changes', ascending=False)
        st.dataframe(activity_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("No units with price changes found")

# Tab 6: AI Recommendations
with tab6:
    st.subheader("🤖 AI-Powered Unit Recommendations (for Snapshot Date)")
    
    st.markdown("""
    Get personalized recommendations for specific units available on **{}** based on market trends, price history, 
    and leasing patterns. Select a unit to receive actionable insights.
    """.format(snapshot_date.strftime('%B %d, %Y')))
    
    # Unit selector
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by apartment type first
        apt_types = sorted(snapshot_df['apartment_type'].unique())
        selected_type = st.selectbox("Select Apartment Type", ["All"] + apt_types, key="rec_type")
    
    with col2:
        # Filter units based on selected type
        if selected_type == "All":
            available_units = sorted(snapshot_df['apartment_id'].unique())
        else:
            available_units = sorted(snapshot_df[snapshot_df['apartment_type'] == selected_type]['apartment_id'].unique())
        
        selected_unit_rec = st.selectbox("Select Specific Unit", 
                                         ["Select a unit..."] + available_units, key="rec_unit")
    
    if selected_unit_rec != "Select a unit...":
        # Get unit data
        unit_data = source_df[df['apartment_id'] == selected_unit_rec].sort_values('date')
        unit_prices = unit_data[unit_data['has_price']]['price_numeric'].values
        
        # Calculate metrics
        days_on_market = (snapshot_date - unit_data['date'].min()).days + 1
        price_changes = 0
        if len(unit_prices) > 1:
            price_changes = len(np.where(np.diff(unit_prices) != 0)[0])
            price_trend = unit_prices[-1] - unit_prices[0]
            price_trend_pct = (price_trend / unit_prices[0]) * 100
        else:
            price_trend = 0
            price_trend_pct = 0
        
        # Generate recommendation
        st.markdown("---")
        st.markdown(f"### 📊 Analysis for Unit {selected_unit_rec}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Days on Market", f"{days_on_market}")
        with col2:
            if len(unit_prices) > 0:
                st.metric("Current Price", f"${unit_prices[-1]:,.0f}", 
                         f"{price_trend_pct:+.1f}%" if price_trend != 0 else None)
            else:
                st.metric("Current Price", "Call for pricing")
        with col3:
            st.metric("Price Changes", f"{price_changes}")
        
        # Detailed recommendation
        st.markdown("### 🎯 Recommendation")
        
        # Decision logic
        if days_on_market > 90 and price_changes >= 5:
            recommendation = "STRONG BUY - LEASE NOW"
            
            st.markdown(f"""
            <div class="success-box">
            <h4>✅ {recommendation}</h4>
            
            <strong>Key Factors:</strong>
            <ul>
            <li>This unit has been on the market for {days_on_market} days (well above average)</li>
            <li>Price has been reduced {price_changes} times, showing landlord flexibility</li>
            <li>Maximum negotiation leverage due to extended vacancy</li>
            <li>Risk of missing out if you wait longer</li>
            </ul>
            
            <strong>Action Items:</strong>
            <ol>
            <li>Offer ${int(unit_prices[-1] * 0.95):,} - ${int(unit_prices[-1] * 0.97):,} (3-5% below asking)</li>
            <li>Request 1-2 months free rent or reduced security deposit</li>
            <li>Ask for parking/storage inclusion</li>
            <li>Lock in a longer lease term for additional discount</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
        elif days_on_market > 60 and price_changes >= 3:
            recommendation = "GOOD OPPORTUNITY - ACT SOON"
            
            st.markdown(f"""
            <div class="recommendation-box">
            <h4>👍 {recommendation}</h4>
            
            <strong>Key Factors:</strong>
            <ul>
            <li>Unit has been available for {days_on_market} days</li>
            <li>Multiple price reductions ({price_changes}) indicate motivation to lease</li>
            <li>Good negotiation position available</li>
            </ul>
            
            <strong>Negotiation Strategy:</strong>
            <ul>
            <li>Target price: ${int(unit_prices[-1] * 0.97):,} - ${int(unit_prices[-1] * 0.99):,}</li>
            <li>Request one month free rent or waived fees</li>
            <li>Consider acting within 1-2 weeks</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif days_on_market < 30:
            recommendation = "WAIT - TOO EARLY"
            
            st.markdown(f"""
            <div class="warning-box">
            <h4>⏳ {recommendation}</h4>
            
            <strong>Key Factors:</strong>
            <ul>
            <li>Recently listed ({days_on_market} days ago)</li>
            <li>Limited price history for negotiation</li>
            <li>Landlord unlikely to negotiate significantly</li>
            </ul>
            
            <strong>Recommended Approach:</strong>
            <ul>
            <li>Monitor for 2-3 more weeks</li>
            <li>Watch for price reductions</li>
            <li>Consider if price drops below ${int(unit_prices[-1] * 0.95):,} if price is available</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            recommendation = "MODERATE - CONSIDER YOUR NEEDS"
            
            st.markdown(f"""
            <div class="recommendation-box">
            <h4>🤔 {recommendation}</h4>
            
            <strong>Key Factors:</strong>
            <ul>
            <li>Market time: {days_on_market} days</li>
            <li>Price adjustments: {price_changes}</li>
            <li>Some negotiation room available</li>
            </ul>
            
            <strong>Suggested Approach:</strong>
            <ul>
            <li>If you need to move soon, negotiate for small concessions</li>
            <li>If flexible, wait 2-3 weeks for potential price drops</li>
            {"<li>Target price: $" + f"{int(unit_prices[-1] * 0.98):,}" + "</li>" if len(unit_prices) > 0 else ""}
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Similar units comparison
        st.markdown("### 🔍 Similar Units Comparison")
        
        similar_units = snapshot_df[(snapshot_df['apartment_type'] == unit_data.iloc[0]['apartment_type']) & 
                                       (snapshot_df['apartment_id'] != selected_unit_rec)]
        
        if len(similar_units) > 0:
            comparison_data = []
            for apt_id in similar_units['apartment_id'].unique()[:5]:  # Top 5 similar
                apt_data = source_df[source_df['apartment_id'] == apt_id]
                apt_prices = apt_data[apt_data['has_price']]['price_numeric'].values
                
                comparison_data.append({
                    'Unit': apt_id,
                    'Floor': apt_data.iloc[0]['floor'],
                    'Current Price': f"${apt_prices[-1]:,.0f}" if len(apt_prices) > 0 else "Call",
                    'Days on Market': (snapshot_date - apt_data['date'].min()).days + 1,
                    'Price Changes': len(np.where(np.diff(apt_prices) != 0)[0]) if len(apt_prices) > 1 else 0
                })
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("No similar units currently available")
    
    # Market timing advice
    st.markdown("---")
    st.subheader("📅 General Market Timing Advice")
    
    current_month = snapshot_date.month
    season_advice = {
        (3, 4, 5): ("Spring", "High demand season. Prices typically peak. Act quickly on good deals."),
        (6, 7, 8): ("Summer", "Peak moving season. Most inventory but also highest prices."),
        (9, 10, 11): ("Fall", "Cooling market. Better negotiation opportunities emerging."),
        (12, 1, 2): ("Winter", "Best deals available. Lowest demand gives maximum negotiation power.")
    }
    
    for months, (season, advice) in season_advice.items():
        if current_month in months:
            st.info(f"**Current Season: {season}** - {advice}")
            break

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <small>Dashboard created for Residences at West Edge market analysis | Data updated through {}</small>
</div>
""".format(latest_date.strftime("%B %d, %Y")), unsafe_allow_html=True)