# For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.

# 2026-01-11 02:19:05.148 Please replace `use_container_width` with `width`.
use_container_width=True
use_container_width=True

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
import historical_dummy_data_generator
# import css_renderer
from assets.css_presets import html_sidebar, html_header, html_sidebar_clear_filters_btn, html_sidebar_nav_link
# Configuration

API_KEY = ""

# Try Streamlit secrets first (for cloud deployment), then fall back to local key.py
try:
    import streamlit as st
    API_KEY = st.secrets["OR_KEY"]
except Exception:
    print("No Streamlit secrets found, trying local key.py...")
    try:
        import key
        API_KEY = key.OR_key
    except ImportError:
        print("No API key found. Set st.secrets['OR_KEY'] or create key.py")

def initialize_ai_client(API_KEY=API_KEY):
    """Initialize AI client for chart descriptions"""
    try:
        from openai import OpenAI
        # API_KEY = ""
        return OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
    except Exception as e:
        st.error(f"Failed to initialize AI client: {e}")
        return None

def generate_chart_description(category_data, total_revenue, ai_client):
    """Generate AI description for category pie chart"""
    if not ai_client:
        return "AI description unavailable - client not initialized."
    
    # Prepare data for AI analysis
    categories = category_data['customer_category'].tolist()
    amounts = category_data['sale_amount'].tolist()
    percentages = [(amount/total_revenue)*100 for amount in amounts]
    
    # Find key insights
    top_category = categories[amounts.index(max(amounts))]
    top_amount = max(amounts)
    top_percentage = max(percentages)
    
    lowest_category = categories[amounts.index(min(amounts))]
    lowest_amount = min(amounts)
    lowest_percentage = min(percentages)
    
    # Create detailed prompt for AI
    prompt = f"""
Analyze this customer category revenue distribution and provide business insights:

**Revenue Breakdown:**
Total Revenue: ${total_revenue:,.0f}

**Category Performance:**
"""
    
    for cat, amount, pct in zip(categories, amounts, percentages):
        prompt += f"- {cat}: ${amount:,.0f} ({pct:.1f}%)\n"
    
    prompt += f"""

**Key Metrics:**
- Strongest Category: {top_category} (${top_amount:,.0f}, {top_percentage:.1f}%)
- Weakest Category: {lowest_category} (${lowest_amount:,.0f}, {lowest_percentage:.1f}%)
- Number of Categories: {len(categories)}

Please provide:
1. **Revenue Analysis** - What does this distribution tell us about our business?
2. **Category Performance** - Which categories are over/under-performing?
3. **Business Risks** - Any concentration risks or concerns?
4. **Strategic Recommendations** - 2-3 actionable suggestions for growth
5. **Market Opportunities** - Potential areas for expansion

Keep it professional, data-driven, and actionable. Use specific numbers from the data.
Format with clear headings and bullet points for readability.
"""
    try:
        response = ai_client.chat.completions.create(
            model="mistralai/devstral-small:free",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a senior business analyst providing insights on revenue data. Be specific, professional, and actionable."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=900
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating AI description: {str(e)}"


def generate_business_journey(df_data, ai_client):
    """Generate comprehensive business journey analysis"""
    if not ai_client:
        return "AI description unavailable - client not initialized."

    # Calculate date range
    date_range_days = (df_data['sale_date'].max() - df_data['sale_date'].min()).days
    years_covered = date_range_days / 365.25

    # Monthly revenue trends
    monthly_revenue = df_data.groupby(df_data['sale_date'].dt.to_period('M'))['sale_amount'].sum()
    avg_monthly_revenue = monthly_revenue.mean()
    peak_month = monthly_revenue.idxmax()
    peak_revenue = monthly_revenue.max()
    lowest_month = monthly_revenue.idxmin()
    lowest_revenue = monthly_revenue.min()

    # Growth metrics
    if len(monthly_revenue) > 1:
        first_half_avg = monthly_revenue[:len(monthly_revenue)//2].mean()
        second_half_avg = monthly_revenue[len(monthly_revenue)//2:].mean()
        growth_rate = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
    else:
        growth_rate = 0

    # Customer insights
    top_customers = df_data.groupby('customer_name')['sale_amount'].sum().nlargest(5)
    total_customers = df_data['customer_name'].nunique()
    top_5_contribution = (top_customers.sum() / df_data['sale_amount'].sum() * 100)

    # Product insights
    top_products = df_data.groupby('product_name')['sale_amount'].sum().nlargest(3)

    # Regional insights
    top_regions = df_data.groupby('warehouse_region')['sale_amount'].sum().nlargest(3)

    # Create comprehensive prompt
    prompt = f"""
Analyze this business performance over the past {years_covered:.1f} years and provide a comprehensive business journey narrative:

**Overall Performance:**
- Total Revenue: ${df_data['sale_amount'].sum():,.0f}
- Average Monthly Revenue: ${avg_monthly_revenue:,.0f}
- Total Transactions: {len(df_data):,}
- Unique Customers: {total_customers}

**Growth Trends:**
- Period-over-period growth: {growth_rate:+.1f}%
- Peak Month: {peak_month} (${peak_revenue:,.0f})
- Lowest Month: {lowest_month} (${lowest_revenue:,.0f})

**Top 5 Customers (contributing {top_5_contribution:.1f}% of revenue):**
{chr(10).join([f"- {name}: ${amount:,.0f}" for name, amount in top_customers.items()])}

**Top 3 Products:**
{chr(10).join([f"- {name}: ${amount:,.0f}" for name, amount in top_products.items()])}

**Top 3 Regions:**
{chr(10).join([f"- {name}: ${amount:,.0f}" for name, amount in top_regions.items()])}

Please provide a comprehensive business journey analysis including:
1. **Journey Overview** - Summarize the {years_covered:.1f}-year business trajectory
2. **Key Milestones** - Highlight peak performance periods and what likely drove them
3. **Challenges Faced** - Identify low points and potential causes
4. **Customer Dynamics** - Analyze customer concentration and loyalty patterns
5. **Growth Drivers** - What products, regions, or segments drove growth
6. **Future Outlook** - Based on trends, what should the business focus on next
7. **Strategic Recommendations** - 3-5 actionable insights for continued growth

Make it engaging, data-driven, and tell a compelling story of the business journey.
Format with clear headings and bullet points for readability.
"""

    try:
        response = ai_client.chat.completions.create(
            model="mistralai/devstral-small:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior business strategist analyzing multi-year business performance. Provide insights that are specific, actionable, and tell a compelling story."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating business journey: {str(e)}"


def create_enhanced_pie_chart(category_dist, is_full_width=True):
    """Create pie chart with dynamic sizing"""
    
    fig_category = px.pie(
        category_dist, 
        values='sale_amount', 
        names='customer_category',
        title='Revenue by Customer Category',
        height=500 if is_full_width else 450,
        color_discrete_sequence=['#418FDE', '#FF6B35', '#1F2B3A', '#8BB8E8', '#F4F6FB', '#E0E7F1']
    )

    fig_category.update_layout(
        title=dict(
            text='Revenue by Customer Category',
            font=dict(size=20 if is_full_width else 18, color='#1F2B3A', family='Segoe UI'),
            x=0.5 if is_full_width else 0.02
        ),
        plot_bgcolor='rgba(244, 246, 251, 0.9)',
        paper_bgcolor='rgba(244, 246, 251, 0.9)',
        font=dict(family="Segoe UI", size=12, color="#1F2B3A"),
        margin=dict(t=80, r=40, l=40, b=40),
        hoverlabel=dict(
            bgcolor="#1F2B3A",
            font_size=13,
            font_family="Segoe UI",
            font_color="white"
        ),
        legend=dict(
            font=dict(color='#1F2B3A', size=12 if is_full_width else 11),
            bgcolor='rgba(244, 246, 251, 0.8)',
            bordercolor='rgba(65, 143, 222, 0.2)',
            borderwidth=1,
            orientation="h" if is_full_width else "v",
            yanchor="bottom" if is_full_width else "top",
            y=-0.2 if is_full_width else 1,
            xanchor="center" if is_full_width else "left",
            x=0.5 if is_full_width else 1.02
        ),
        showlegend=True
    )

    fig_category.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=12 if is_full_width else 11,
        textfont_color='white',
        marker=dict(line=dict(color='white', width=2)),
        hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
    )
    
    return fig_category

def create_smooth_transition_css():

    """CSS for smooth transitions and scrollable description box"""
    return """
    <style>
    /* Smooth transition animations */
    .chart-container {
        transition: all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform-origin: center;
    }
    
    .description-panel {
        transition: all 0.6s ease-in-out;
        opacity: 0;
        transform: translateX(50px);
        animation: slideInRight 0.8s ease-out forwards;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .describe-button {
        background: linear-gradient(135deg, #418FDE, #66e6ff) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(65, 143, 222, 0.3) !important;
    }
    
    .describe-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(65, 143, 222, 0.4) !important;
    }
    
    /* Scrollable AI insights container */
    .ai-insights-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        border-left: 4px solid #418FDE;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        height: 400px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    
    .ai-insights-header {
        background: linear-gradient(135deg, #418FDE, #66e6ff);
        color: white;
        padding: 1rem;
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        text-align: center;
    }
    
    .ai-insights-content {
        flex: 1;
        overflow-y: auto;
        padding: 1.5rem;
        background: white;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Custom scrollbar */
    .ai-insights-content::-webkit-scrollbar {
        width: 6px;
    }
    
    .ai-insights-content::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    
    .ai-insights-content::-webkit-scrollbar-thumb {
        background: #418FDE;
        border-radius: 3px;
    }
    
    .ai-insights-content::-webkit-scrollbar-thumb:hover {
        background: #66e6ff;
    }
    
    .ai-insights-content h4 {
        color: #1F2B3A;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #418FDE;
        padding-bottom: 0.25rem;
    }
    
    .ai-insights-content h4:first-child {
        margin-top: 0;
    }
    
    .ai-insights-content ul {
        margin: 0.5rem 0;
        padding-left: 1.2rem;
    }
    
    .ai-insights-content li {
        margin-bottom: 0.3rem;
        color: #374151;
    }
    
    .ai-insights-content strong {
        color: #1F2B3A;
    }
    
    /* Quick stats styling */
    .quick-stats-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .quick-stats-container h4 {
        color: #1F2B3A;
        margin-bottom: 0.8rem;
        font-size: 1rem;
        text-align: center;
    }
    </style>
    """

@st.cache_resource
def get_cached_ai_client():
    """Cache AI client to avoid re-initialization"""
    return initialize_ai_client()

st.set_page_config(page_title="Maize Distribution Analytics", layout="wide")

df = pd.read_csv('partial_csv.csv')

df['sale_date'] = pd.to_datetime(df['sale_date'])
df['year'] = df['sale_date'].dt.year
df['month'] = df['sale_date'].dt.month

df_filtered = df.copy()

# Get date range from data
min_date = df['sale_date'].min().date()
max_date = df['sale_date'].max().date()

st.sidebar.title("Filters")

# Date filtering section
st.sidebar.subheader("📅 Date Range")

# # Quick preset buttons in columns
# preset_col1, preset_col2 = st.sidebar.columns(2)

# with preset_col1:
#     if st.button("Last 30 Days", key="last_30", use_container_width=True):
#         st.session_state.date_start = max_date - timedelta(days=30)
#         st.session_state.date_end = max_date
#         st.rerun()
    
#     if st.button("Last 6 Months", key="last_6m", use_container_width=True):
#         st.session_state.date_start = max_date - timedelta(days=180)
#         st.session_state.date_end = max_date
#         st.rerun()

# with preset_col2:
#     if st.button("Last 90 Days", key="last_90", use_container_width=True):
#         st.session_state.date_start = max_date - timedelta(days=90)
#         st.session_state.date_end = max_date
#         st.rerun()
    
#     if st.button("All Time", key="all_time", use_container_width=True):
#         st.session_state.date_start = min_date
#         st.session_state.date_end = max_date
#         st.rerun()

# Quick preset dropdown
def handle_date_preset_change():
    """Handle date preset selection changes"""
    preset_option = st.session_state.get('date_preset', 'All Time')
    
    if preset_option == 'Last 30 Days':
        st.session_state.date_start = max_date - timedelta(days=30)
        st.session_state.date_end = max_date
    elif preset_option == 'Last 90 Days':
        st.session_state.date_start = max_date - timedelta(days=90)
        st.session_state.date_end = max_date
    elif preset_option == 'Last 6 Months':
        st.session_state.date_start = max_date - timedelta(days=180)
        st.session_state.date_end = max_date
    elif preset_option == 'All Time':
        st.session_state.date_start = min_date
        st.session_state.date_end = max_date
    
    # Reset manual date inputs to match preset
    st.session_state.date_start_input = st.session_state.date_start
    st.session_state.date_end_input = st.session_state.date_end

# Date preset selector
# preset_options = ['Last 30 Days', 'Last 90 Days', 'Last 6 Months', 'All Time']

# # Determine current value (use session state if exists, otherwise default)
# if 'date_preset' in st.session_state:
#     current_preset = st.session_state.date_preset
# else:
#     current_preset = 'All Time'

# selected_preset = st.sidebar.selectbox(
#     "📅 Quick Date Presets:",
#     options=preset_options,
#     value=current_preset,  # Use value instead of index
#     key='date_preset',
#     on_change=handle_date_preset_change
# )

preset_options = ['Last 30 Days', 'Last 90 Days', 'Last 6 Months', 'All Time']

# Determine current index (use session state if exists, otherwise default to 'All Time')
if 'date_preset' in st.session_state:
    # If it's "Custom Range" or any invalid preset, default to "All Time"
    if st.session_state.date_preset in preset_options:
        current_index = preset_options.index(st.session_state.date_preset)
    else:
        current_index = 3  # Default to 'All Time' for "Custom Range" or invalid values
else:
    current_index = 3  # Default to 'All Time'

selected_preset = st.sidebar.selectbox(
    "📅 Quick Date Presets:",
    options=preset_options,
    index=current_index,  # Use index instead of value
    key='date_preset',
    on_change=handle_date_preset_change
)

# Initialize session state for dates if not exists
if 'date_start' not in st.session_state:
    st.session_state.date_start = min_date # max_date - timedelta(days=90)  # Default to last 90 days
if 'date_end' not in st.session_state:
    st.session_state.date_end = max_date

# Manual date range selector
st.sidebar.markdown("**📅 Or set custom dates:**")
date_col1, date_col2 = st.sidebar.columns(2)

def handle_manual_date_change():
    """Handle manual date input changes - reset preset to None"""
    if 'date_start_input' in st.session_state and 'date_end_input' in st.session_state:
        # Check if dates were manually changed
        manual_start = st.session_state.date_start_input
        manual_end = st.session_state.date_end_input
        
        if (manual_start != st.session_state.date_start or 
            manual_end != st.session_state.date_end):
            # Reset preset selection
            st.session_state.date_preset = 'Custom Range'

with date_col1:
    start_date = st.date_input(
        "From",
        value=st.session_state.date_start,
        min_value=min_date,
        max_value=max_date,
        key="date_start_input",
        on_change=handle_manual_date_change
    )

with date_col2:
    end_date = st.date_input(
        "To", 
        value=st.session_state.date_end,
        min_value=min_date,
        max_value=max_date,
        key="date_end_input",
        on_change=handle_manual_date_change
    )

# Show current selection type
if st.session_state.get('date_preset') in ['Last 30 Days', 'Last 90 Days', 'Last 6 Months', 'All Time']:
    st.sidebar.info(f"📊 **Using Preset:** {st.session_state.date_preset}")
else:
    st.sidebar.info(f"📊 **Using Custom Range:** {start_date} to {end_date}")

# Apply date filtering
df_filtered = df[
    (df['sale_date'].dt.date >= start_date) & 
    (df['sale_date'].dt.date <= end_date)
].copy()

# Display current selection
st.sidebar.info(f"📊 **Selected Period:**\n{start_date} to \n{end_date}\n\n**Records:** {len(df_filtered):,}")

# Clear All Filters button
if st.sidebar.button("🔄 Clear All Filters", use_container_width=True):
    # Clear all filter-related session state
    keys_to_clear = [
        'date_preset', 'date_start', 'date_end',
        'date_start_input', 'date_end_input',
        'customer_filter', 'category_filter',
        'region_filter', 'product_filter', 'search_input'
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()

# Other filters (move this BEFORE the charts)
customer_options = sorted(df_filtered['customer_name'].unique())  # Use df_filtered here
selected_customers = st.sidebar.multiselect('Select Customers', customer_options, default=[], key='customer_filter')

category_options = sorted(df_filtered['customer_category'].unique())  # Use df_filtered here
selected_categories = st.sidebar.selectbox('Select Customer Category', ['All'] + list(category_options), key='category_filter')

region_options = sorted(df_filtered['warehouse_region'].unique())  # Use df_filtered here
selected_region = st.sidebar.selectbox('Select Region', ['All'] + list(region_options), key='region_filter')

product_options = sorted(df_filtered['product_name'].unique())  # Use df_filtered here
selected_product = st.sidebar.selectbox('Select Product', ['All'] + list(product_options), key='product_filter')

# ⭐ ADD THIS: Apply the additional filters to df_filtered
if selected_customers:
    df_filtered = df_filtered[df_filtered['customer_name'].isin(selected_customers)]

if selected_categories != 'All':
    df_filtered = df_filtered[df_filtered['customer_category'] == selected_categories]

if selected_region != 'All':
    df_filtered = df_filtered[df_filtered['warehouse_region'] == selected_region]

if selected_product != 'All':
    df_filtered = df_filtered[df_filtered['product_name'] == selected_product]


# html_sidebar_nav_link()
# html_sidebar()
# html_header()
# html_sidebar_clear_filters_btn()
# html_main_page()

# Dashboard header
st.title("🌽 Maize Distribution Analytics")
st.subheader("Basic Insights Dashboard")
st.markdown("---")

all_year_options = sorted(df['year'].unique())[-1:]

top_most_col1, top_most_col2, top_most_col3, top_most_col4 = st.columns([2,2,2,1])

with top_most_col1:
        st.markdown(f"Metrics as of: {max_date}")

col1, col2, col3, col4 = st.columns(4)

# Calculate headline metrics (all-time, unfiltered)
total_revenue = df['sale_amount'].sum()
total_volume = df['final_tons_sold'].sum()
avg_order_size = df['final_tons_sold'].mean()
total_customers = df['customer_name'].nunique()

# Calculate last 30 days for delta comparison
current_date = df['sale_date'].max()
last_30_days_start = current_date - pd.Timedelta(days=30)
previous_30_days_start = last_30_days_start - pd.Timedelta(days=30)

# Current period (last 30 days)
current_period = df[df['sale_date'] >= last_30_days_start]
current_revenue = current_period['sale_amount'].sum()
current_volume = current_period['final_tons_sold'].sum()
current_avg_order = current_period['final_tons_sold'].mean()
current_customers = current_period['customer_name'].nunique()

# Previous period (30 days before that)
previous_period = df[(df['sale_date'] >= previous_30_days_start) & (df['sale_date'] < last_30_days_start)]
previous_revenue = previous_period['sale_amount'].sum()
previous_volume = previous_period['final_tons_sold'].sum()
previous_avg_order = previous_period['final_tons_sold'].mean()
previous_customers = previous_period['customer_name'].nunique()

# Calculate percentage changes
revenue_change = ((current_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0
volume_change = ((current_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0
avg_order_change = ((current_avg_order - previous_avg_order) / previous_avg_order * 100) if previous_avg_order > 0 else 0
customer_change = ((current_customers - previous_customers) / previous_customers * 100) if previous_customers > 0 else 0

with col1:
    st.metric(
        "Total Revenue",
        value=f"${total_revenue:,.0f}",
        delta=f"{revenue_change:+.1f}% vs last 30 days"
    )
with col2:
    st.metric(
        "Total Volume (Tons)",
        f"{total_volume:,.0f}",
        delta=f"{volume_change:+.1f}% vs last 30 days"
    )
with col3:
    st.metric(
        "Avg Order Size (Tons)",
        f"{avg_order_size:.1f}",
        delta=f"{avg_order_change:+.1f}% vs last 30 days"
    )
with col4:
    st.metric(
        "Total Customers",
        total_customers,
        delta=f"{customer_change:+.1f}% vs last 30 days"
    )

# Add spacing after metrics
st.markdown("---")

# Revenue Trend - Direct styling approach
st.subheader("📈 Revenue Trend")

monthly_revenue = df_filtered.groupby(df_filtered['sale_date'].dt.strftime('%Y-%m'))[['sale_amount']].sum().reset_index()

# Create enhanced chart with gradient and better styling
fig_revenue = go.Figure()

# Add area fill under the line for visual impact
fig_revenue.add_trace(go.Scatter(
    x=monthly_revenue['sale_date'],
    y=monthly_revenue['sale_amount'],
    mode='lines+markers',
    name='Revenue',
    line=dict(
        color='#FF6B35',  # Vibrant orange-red for contrast
        width=4,
        shape='spline'  # Smooth curves
    ),
    marker=dict(
        size=12, 
        color='#FF6B35',
        line=dict(color='white', width=2),
        symbol='circle'
    ),
    fill='tonexty',
    fillcolor='rgba(255, 107, 53, 0.1)',  # Light fill under line
    hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
))

# Add invisible trace at y=0 for area fill
fig_revenue.add_trace(go.Scatter(
    x=monthly_revenue['sale_date'],
    y=[0] * len(monthly_revenue),
    mode='lines',
    line=dict(color='rgba(0,0,0,0)'),
    showlegend=False,
    hoverinfo='skip'
))

# Add value annotations on peaks
max_revenue_idx = monthly_revenue['sale_amount'].idxmax()
max_revenue_point = monthly_revenue.iloc[max_revenue_idx]

fig_revenue.add_annotation(
    x=max_revenue_point['sale_date'],
    y=max_revenue_point['sale_amount'],
    text=f"Peak: ${max_revenue_point['sale_amount']:,.0f}",
    showarrow=True,
    arrowhead=2,
    arrowcolor='#1F2B3A',
    arrowwidth=2,
    bgcolor='#1F2B3A',
    bordercolor='#FF6B35',
    borderwidth=2,
    font=dict(color='white', size=12, family='Segoe UI')
)

fig_revenue.update_layout(
    title=dict(
        text='Monthly Revenue Performance',
        font=dict(size=20, color='#1F2B3A', family='Segoe UI'),
        x=0.02 # Left-aligned title
    ),
    xaxis=dict(
        title=dict(text='Period', font=dict(size=14, color='#1F2B3A')),
        tickfont=dict(size=12, color='#1F2B3A'),
        gridcolor='rgba(31, 43, 58, 0.1)',
        linecolor='#1F2B3A',
        showgrid=True,
        ticklen=8,          # Length of tick marks
        tickwidth=1,        # Width of tick marks
        
    ),
    yaxis=dict(
        title=dict(text='Revenue ($)', font=dict(size=14, color='#1F2B3A')),
        tickfont=dict(size=12, color='#1F2B3A'),
        tickformat='$,.0f',
        gridcolor='rgba(31, 43, 58, 0.1)',
        linecolor='#1F2B3A',
        showgrid=True,
        # tickmode='linear',  # Ensures consistent spacing
        # dtick='auto',       # Auto-spacing for ticks
        ticklen=8,          # Length of tick marks
        tickwidth=1,        # Width of tick marks
        tickcolor='#1F2B3A' # Color of tick marks
    ),
    plot_bgcolor='rgba(65, 143, 222, 0.05)',
    paper_bgcolor='rgba(244, 246, 251, 0.9)',
    height=500,
    margin=dict(t=100, r=20, l=60, b=20),
    font=dict(family="Segoe UI", size=12, color="#1F2B3A"),
    hoverlabel=dict(
        bgcolor="#1F2B3A",
        font_size=14,
        font_family="Segoe UI",
        font_color="white",
        bordercolor='#FF6B35'
    ),
    showlegend=False
)
# Display the chart without any wrapper
st.plotly_chart(fig_revenue, use_container_width=True)


# pie_col1, pie_col2 = st.columns([1, 1])

# category_dist = df_filtered.groupby('customer_category')['sale_amount'].sum().reset_index()

# with pie_col1:

#     fig_category = px.pie(
#         category_dist, 
#         values='sale_amount', 
#         names='customer_category',
#         title='Revenue by Customer Category',
#         height=450,
#         color_discrete_sequence=['#418FDE', '#FF6B35', '#1F2B3A', '#8BB8E8', '#F4F6FB', '#E0E7F1']
#     )

#     fig_category.update_layout(
#         title=dict(
#             text='Revenue by Customer Category',
#             font=dict(size=18, color='#1F2B3A', family='Segoe UI'),
#             x=0.02
#         ),
#         plot_bgcolor='rgba(244, 246, 251, 0.9)',
#         paper_bgcolor='rgba(244, 246, 251, 0.9)',
#         font=dict(family="Segoe UI", size=12, color="#1F2B3A"),
#         margin=dict(t=60, r=40, l=40, b=40),
#         hoverlabel=dict(
#             bgcolor="#1F2B3A",
#             font_size=13,
#             font_family="Segoe UI",
#             font_color="white"
#         ),
#         legend=dict(
#             font=dict(color='#1F2B3A', size=11),
#             bgcolor='rgba(244, 246, 251, 0.8)',
#             bordercolor='rgba(65, 143, 222, 0.2)',
#             borderwidth=1
#         )
#     )

#     fig_category.update_traces(
#         textposition='inside',
#         textinfo='percent+label',
#         textfont_size=11,
#         textfont_color='white',
#         marker=dict(line=dict(color='white', width=2)),
#         hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
#     )

#     # Add container class for styling
#     st.markdown('<div class="pie-chart-container">', unsafe_allow_html=True)
#     st.plotly_chart(fig_category, use_container_width=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# with pie_col2:
#     # Create a simple text box for additional information
#     st.markdown("""
#         <div class="info-box">
#             <h3>Customer Category Insights</h3>
#             <p>This pie chart shows the distribution of revenue across different customer categories. 
#             Use this information to identify key customer segments and tailor your marketing strategies accordingly.</p>
#         </div>
#     """, unsafe_allow_html=True)

# ====== PIE CHARTS - ALL THREE IN ONE ROW ======

st.subheader("📊 Distribution Analysis")

# Prepare data for all three pie charts
category_dist = df_filtered.groupby('customer_category')['sale_amount'].sum().reset_index()
region_dist = df_filtered.groupby('warehouse_region')['sale_amount'].sum().reset_index()
product_mix = df_filtered.groupby('product_name')['final_tons_sold'].sum().reset_index()

# Create three columns for pie charts
pie_col1, pie_col2, pie_col3 = st.columns(3)

# Chart 1: Customer Category
with pie_col1:
    st.markdown("**Revenue by Customer Category**")
    fig_category = px.pie(
        category_dist,
        values='sale_amount',
        names='customer_category',
        height=400,
        color_discrete_sequence=['#418FDE', '#FF6B35', '#1F2B3A', '#8BB8E8']
    )
    fig_category.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=10,
        marker=dict(line=dict(color='white', width=2))
    )
    fig_category.update_layout(
        margin=dict(t=20, r=20, l=20, b=20),
        showlegend=True,
        legend=dict(font=dict(size=9))
    )
    st.plotly_chart(fig_category, use_container_width=True)

# Chart 2: Warehouse Region
with pie_col2:
    st.markdown("**Revenue by Warehouse Region**")
    fig_region = px.pie(
        region_dist,
        values='sale_amount',
        names='warehouse_region',
        height=400,
        color_discrete_sequence=['#FF6B35', '#418FDE', '#1F2B3A', '#8BB8E8', '#F4F6FB']
    )
    fig_region.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=10,
        marker=dict(line=dict(color='white', width=2))
    )
    fig_region.update_layout(
        margin=dict(t=20, r=20, l=20, b=20),
        showlegend=True,
        legend=dict(font=dict(size=9))
    )
    st.plotly_chart(fig_region, use_container_width=True)

# Chart 3: Product Mix
with pie_col3:
    st.markdown("**Sales Volume by Product Type**")
    fig_product = px.pie(
        product_mix,
        values='final_tons_sold',
        names='product_name',
        height=400,
        color_discrete_sequence=['#1F2B3A', '#FF6B35', '#418FDE', '#8BB8E8']
    )
    fig_product.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=10,
        marker=dict(line=dict(color='white', width=2))
    )
    fig_product.update_layout(
        margin=dict(t=20, r=20, l=20, b=20),
        showlegend=True,
        legend=dict(font=dict(size=9))
    )
    st.plotly_chart(fig_product, use_container_width=True)

st.markdown("---")

# ====== DESCRIBE BUSINESS JOURNEY SECTION ======
st.subheader("📖 Business Journey Analysis")

# Initialize session state
if 'show_journey' not in st.session_state:
    st.session_state.show_journey = False

# Button to trigger analysis
col_button = st.columns([2, 1, 2])
with col_button[1]:
    if st.button("🧠 Analyze Business Journey", key="describe_journey", use_container_width=True, type="primary"):
        st.session_state.show_journey = not st.session_state.show_journey
        st.rerun()

# Show journey analysis if toggled
if st.session_state.show_journey:
    with st.spinner("🔍 Analyzing your business journey..."):
        ai_client = get_cached_ai_client()
        if ai_client:
            journey_analysis = generate_business_journey(df, ai_client)

            st.markdown(f"""
            <div style="background-color: #f4f6fb; border-left: 4px solid #418FDE; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #418FDE; margin-top: 0;">🤖 AI-Generated Business Journey Insights</h3>
                <div style="color: #1F2B3A; line-height: 1.8;">
                    {journey_analysis}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Unable to generate business journey analysis. Please check AI client configuration.")

st.markdown("---")
st.subheader("🔍 Search by Customer Name")

search_value = st.text_input(f"Enter Customer Name:", key='search_input')

if search_value:
    df_filtered = df_filtered[df_filtered['customer_name'].str.contains(search_value, case=False)]

# Customer table with filtered data
st.markdown("---")
st.subheader("Full Data Table")

# Create filtered customer table using the same filtered dataset
customer_table = df_filtered.groupby(['customer_name', 'customer_category', 'warehouse_region']).\
    agg({
        'sale_amount': 'sum',
        'final_tons_sold': 'sum'
    }).reset_index()

customer_table = customer_table.sort_values('sale_amount', ascending=False)
customer_table['sale_amount'] = customer_table['sale_amount'].round(2)
customer_table['final_tons_sold'] = customer_table['final_tons_sold'].round(2)

# Format revenue as currency
customer_table['sale_amount'] = customer_table['sale_amount'].apply(lambda x: f"${x:,.2f}")

# Rename columns for better presentation
customer_table.columns = ['Customer Name', 'Category', 'Region', 'Revenue', 'Volume (Tons)']

st.dataframe(
    customer_table.style\
    .apply(lambda x: ['background-color: #ffffff' if i % 2 == 0 else 'background-color: #dbe7f3' for i in range(len(x))], axis=1)\
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#418FDE'), ('color', 'white'), ('font-weight', 'bold')]},
        {'selector': 'thead th:first-child', 'props': [('background-color', '#418FDE'), ('color', 'white')]},
        {'selector': 'td:first-child', 'props': [('font-weight', 'bold')]}
    ]),
    use_container_width=True 
)

# Add a note about the data
st.sidebar.markdown("---")
st.sidebar.markdown("ℹ️ **Note:** This dashboard uses dummy data for demonstration purposes.")