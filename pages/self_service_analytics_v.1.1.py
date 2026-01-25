"""
Self-Service Analytics Page
Build your own visualizations with AI-powered insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path to import ai_insights
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import AI insights module
try:
    from ai_insights import AIInsightsAnalyzer, create_insights_analyzer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ============== CONFIGURATION ==============
st.set_page_config(
    page_title="Self-Service Analytics | Maize Distribution",
    page_icon="🎨",
    layout="wide"
)

# ============== API KEY SETUP ==============
API_KEY = ""
try:
    API_KEY = st.secrets.get("OR_KEY", "")
except Exception:
    pass

if not API_KEY:
    try:
        import key
        API_KEY = key.OR_key
    except ImportError:
        pass

# ============== STYLING ==============
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #1F2B3A 0%, #2d4a6f 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Chart builder card */
    .chart-builder-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .chart-builder-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #418FDE;
    }
    
    .chart-builder-header h3 {
        margin: 0;
        color: #1F2B3A;
        font-size: 1.2rem;
    }
    
    /* AI Insight box */
    .ai-insight-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #418FDE;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
    }
    
    .ai-insight-box h4 {
        color: #1F2B3A;
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .ai-insight-content {
        color: #374151;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-ready {
        background: #dcfce7;
        color: #166534;
    }
    
    .status-empty {
        background: #fef3c7;
        color: #92400e;
    }
    
    /* Instructions panel */
    .instructions-panel {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px dashed #cbd5e1;
    }
    
    .instructions-panel h4 {
        margin: 0 0 0.5rem 0;
        color: #1F2B3A;
    }
    
    .instructions-panel ol {
        margin: 0;
        padding-left: 1.25rem;
        color: #64748b;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chart container */
    .chart-display {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ============== DATA LOADING ==============
@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv('partial_csv.csv')
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['year'] = df['sale_date'].dt.year
        df['month'] = df['sale_date'].dt.month
        df['year_month'] = df['sale_date'].dt.strftime('%Y-%m')
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found. Please ensure 'partial_csv.csv' exists.")
        return None


@st.cache_resource
def get_ai_analyzer():
    """Get cached AI analyzer instance"""
    if API_KEY and AI_AVAILABLE:
        try:
            return create_insights_analyzer(API_KEY)
        except Exception as e:
            st.warning(f"AI insights unavailable: {e}")
            return None
    return None


# ============== CHART GENERATION ==============
def generate_chart(df: pd.DataFrame, config: Dict) -> tuple:
    """
    Generate a chart based on user configuration
    
    Returns: (plotly_figure, chart_data_for_ai)
    """
    metric = config['metric']
    group_by = config['group_by']
    chart_type = config['chart_type']
    time_period = config['time_period']
    
    # Apply time filter
    df_filtered = apply_time_filter(df.copy(), time_period)
    
    if df_filtered.empty:
        return None, None
    
    # Map metric to column
    metric_mapping = {
        'Revenue': ('sale_amount', 'sum', '$'),
        'Volume (Tons)': ('final_tons_sold', 'sum', ''),
        'Order Count': ('sale_amount', 'count', ''),
        'Avg Order Size': ('final_tons_sold', 'mean', ''),
        'Avg Satisfaction': ('satisfaction_rating', 'mean', '')
    }
    
    col, agg_func, prefix = metric_mapping.get(metric, ('sale_amount', 'sum', '$'))
    
    # Map group_by to column
    group_mapping = {
        'Customer Category': 'customer_category',
        'Region': 'warehouse_region',
        'Product': 'product_name',
        'Customer Size': 'customer_company_size',
        'Customer (Top 10)': 'customer_name',
        'Month': 'year_month'
    }
    
    group_col = group_mapping.get(group_by, 'customer_category')
    
    # Aggregate data
    if agg_func == 'sum':
        chart_df = df_filtered.groupby(group_col)[col].sum().reset_index()
    elif agg_func == 'mean':
        chart_df = df_filtered.groupby(group_col)[col].mean().reset_index()
    else:  # count
        chart_df = df_filtered.groupby(group_col)[col].count().reset_index()
    
    chart_df.columns = ['Category', 'Value']
    
    # Sort and limit for customer view
    if group_by == 'Customer (Top 10)':
        chart_df = chart_df.nlargest(10, 'Value')
    
    # Sort by month if time-based
    if group_by == 'Month':
        chart_df = chart_df.sort_values('Category')
    else:
        chart_df = chart_df.sort_values('Value', ascending=False)
    
    # Color scheme
    colors = ['#418FDE', '#FF6B35', '#1F2B3A', '#8BB8E8', '#F4A460', '#2E8B57', 
              '#9370DB', '#20B2AA', '#FF7F50', '#6495ED']
    
    # Generate chart based on type
    title = f"{metric} by {group_by}"
    
    if chart_type == 'Bar':
        fig = px.bar(
            chart_df, 
            x='Category', 
            y='Value',
            title=title,
            color='Category',
            color_discrete_sequence=colors
        )
        fig.update_layout(showlegend=False)
        
    elif chart_type == 'Horizontal Bar':
        fig = px.bar(
            chart_df, 
            x='Value', 
            y='Category',
            title=title,
            color='Category',
            color_discrete_sequence=colors,
            orientation='h'
        )
        fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        
    elif chart_type == 'Line':
        fig = px.line(
            chart_df, 
            x='Category', 
            y='Value',
            title=title,
            markers=True
        )
        fig.update_traces(line_color='#418FDE', line_width=3, marker_size=10)
        
    elif chart_type == 'Area':
        fig = px.area(
            chart_df, 
            x='Category', 
            y='Value',
            title=title
        )
        fig.update_traces(fillcolor='rgba(65, 143, 222, 0.3)', line_color='#418FDE')
        
    elif chart_type == 'Pie':
        fig = px.pie(
            chart_df, 
            values='Value', 
            names='Category',
            title=title,
            color_discrete_sequence=colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
    elif chart_type == 'Donut':
        fig = px.pie(
            chart_df, 
            values='Value', 
            names='Category',
            title=title,
            hole=0.4,
            color_discrete_sequence=colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
    
    else:  # Default to bar
        fig = px.bar(chart_df, x='Category', y='Value', title=title)
    
    # Apply consistent styling
    fig.update_layout(
        font=dict(family="Segoe UI", size=12, color="#1F2B3A"),
        title=dict(font=dict(size=18, color='#1F2B3A'), x=0.5),
        plot_bgcolor='rgba(248, 250, 252, 0.8)',
        paper_bgcolor='white',
        margin=dict(t=60, r=20, l=20, b=40),
        height=400,
        hoverlabel=dict(
            bgcolor="#1F2B3A",
            font_size=13,
            font_color="white"
        )
    )
    
    # Prepare data for AI analysis
    ai_data = {
        'categories': chart_df['Category'].tolist(),
        'values': chart_df['Value'].tolist(),
        'metric': metric,
        'group_by': group_by,
        'time_period': time_period,
        'total': chart_df['Value'].sum(),
        'average': chart_df['Value'].mean(),
        'max_category': chart_df.loc[chart_df['Value'].idxmax(), 'Category'] if not chart_df.empty else 'N/A',
        'min_category': chart_df.loc[chart_df['Value'].idxmin(), 'Category'] if not chart_df.empty else 'N/A'
    }
    
    return fig, ai_data


def apply_time_filter(df: pd.DataFrame, time_period: str) -> pd.DataFrame:
    """Apply time period filter to dataframe"""
    if df.empty:
        return df
        
    max_date = df['sale_date'].max()
    
    if time_period == 'Last 30 Days':
        start_date = max_date - timedelta(days=30)
    elif time_period == 'Last 90 Days':
        start_date = max_date - timedelta(days=90)
    elif time_period == 'Last 6 Months':
        start_date = max_date - timedelta(days=180)
    elif time_period == 'Year to Date':
        start_date = datetime(max_date.year, 1, 1)
    else:  # All Time
        return df
    
    return df[df['sale_date'] >= start_date]


def get_ai_insight(analyzer, chart_data: Dict, chart_type: str) -> str:
    """Get AI-generated insight for chart data"""
    if not analyzer or not chart_data:
        return None
    
    try:
        # Build a custom prompt for self-service charts
        metric = chart_data.get('metric', 'Value')
        group_by = chart_data.get('group_by', 'Category')
        categories = chart_data.get('categories', [])
        values = chart_data.get('values', [])
        
        if not categories or not values:
            return "No data available for analysis."
        
        # Format data for prompt
        data_pairs = list(zip(categories, values))
        top_item = max(data_pairs, key=lambda x: x[1])
        bottom_item = min(data_pairs, key=lambda x: x[1])
        total = sum(values)
        
        prompt = f"""
Analyze this custom chart and provide 2-3 concise business insights:

**Chart: {metric} by {group_by}**
- Time Period: {chart_data.get('time_period', 'All Time')}
- Total {metric}: {total:,.2f}
- Top Performer: {top_item[0]} ({top_item[1]:,.2f})
- Lowest Performer: {bottom_item[0]} ({bottom_item[1]:,.2f})
- Data Points: {dict(zip(categories[:8], [f'{v:,.2f}' for v in values[:8]]))}

Provide insights about:
1. Key pattern or finding
2. Business implication
3. Quick recommendation

Keep it to 3-4 sentences max. Use one emoji per insight. Be specific with numbers.
"""
        
        return analyzer._call_model(prompt, 'creative')
        
    except Exception as e:
        return f"Analysis unavailable: {str(e)}"


# ============== MAIN PAGE ==============
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Self-Service Analytics</h1>
        <p>Build custom visualizations with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Get AI analyzer
    analyzer = get_ai_analyzer()
    
    # Instructions
    st.markdown("""
    <div class="instructions-panel">
        <h4>📋 How to Use</h4>
        <ol>
            <li>Configure your chart using the dropdowns below</li>
            <li>Click <strong>"Generate Chart"</strong> to create your visualization</li>
            <li>AI will automatically analyze and explain the chart</li>
            <li>Build up to 3 charts per session</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for charts
    if 'charts' not in st.session_state:
        st.session_state.charts = {1: None, 2: None, 3: None}
    if 'chart_configs' not in st.session_state:
        st.session_state.chart_configs = {1: {}, 2: {}, 3: {}}
    if 'ai_insights' not in st.session_state:
        st.session_state.ai_insights = {1: None, 2: None, 3: None}
    
    # Configuration options
    metric_options = ['Revenue', 'Volume (Tons)', 'Order Count', 'Avg Order Size', 'Avg Satisfaction']
    group_options = ['Customer Category', 'Region', 'Product', 'Customer Size', 'Customer (Top 10)', 'Month']
    chart_options = ['Bar', 'Horizontal Bar', 'Line', 'Area', 'Pie', 'Donut']
    time_options = ['All Time', 'Last 30 Days', 'Last 90 Days', 'Last 6 Months', 'Year to Date']
    
    # Create tabs for each chart slot
    tab1, tab2, tab3 = st.tabs(["📊 Chart 1", "📈 Chart 2", "📉 Chart 3"])
    
    tabs = [tab1, tab2, tab3]
    
    for i, tab in enumerate(tabs, 1):
        with tab:
            # Check if chart exists
            has_chart = st.session_state.charts[i] is not None
            
            # Status indicator
            if has_chart:
                st.markdown('<span class="status-badge status-ready">✓ Chart Ready</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-empty">○ Empty Slot</span>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Configuration columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                metric = st.selectbox(
                    "📏 Metric",
                    metric_options,
                    key=f"metric_{i}",
                    help="What do you want to measure?"
                )
            
            with col2:
                group_by = st.selectbox(
                    "📂 Group By",
                    group_options,
                    key=f"group_{i}",
                    help="How do you want to break down the data?"
                )
            
            with col3:
                chart_type = st.selectbox(
                    "📊 Chart Type",
                    chart_options,
                    key=f"chart_type_{i}",
                    help="Choose visualization style"
                )
            
            with col4:
                time_period = st.selectbox(
                    "📅 Time Period",
                    time_options,
                    key=f"time_{i}",
                    help="Filter by time range"
                )
            
            # Action buttons
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
            
            with btn_col1:
                generate_btn = st.button(
                    "🚀 Generate Chart",
                    key=f"generate_{i}",
                    use_container_width=True,
                    type="primary"
                )
            
            with btn_col2:
                clear_btn = st.button(
                    "🗑️ Clear",
                    key=f"clear_{i}",
                    use_container_width=True
                )
            
            # Handle button clicks
            if generate_btn:
                config = {
                    'metric': metric,
                    'group_by': group_by,
                    'chart_type': chart_type,
                    'time_period': time_period
                }
                
                with st.spinner("🔄 Generating chart..."):
                    fig, chart_data = generate_chart(df, config)
                    
                    if fig:
                        st.session_state.charts[i] = fig
                        st.session_state.chart_configs[i] = config
                        
                        # Get AI insight
                        if analyzer:
                            with st.spinner("🧠 AI is analyzing..."):
                                insight = get_ai_insight(analyzer, chart_data, chart_type)
                                st.session_state.ai_insights[i] = insight
                        
                        st.rerun()
                    else:
                        st.error("No data available for this configuration.")
            
            if clear_btn:
                st.session_state.charts[i] = None
                st.session_state.chart_configs[i] = {}
                st.session_state.ai_insights[i] = None
                st.rerun()
            
            st.markdown("---")
            
            # Display chart if exists
            if st.session_state.charts[i]:
                st.plotly_chart(st.session_state.charts[i], use_container_width=True)
                
                # Display AI insight
                insight = st.session_state.ai_insights[i]
                if insight:
                    st.markdown(f"""
                    <div class="ai-insight-box">
                        <h4>🤖 AI Insight</h4>
                        <div class="ai-insight-content">{insight}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif analyzer:
                    st.info("💡 AI insight is being generated...")
                else:
                    st.warning("⚠️ AI insights unavailable - API key not configured")
                
                # Show config summary
                config = st.session_state.chart_configs[i]
                if config:
                    st.caption(f"📋 Config: {config.get('metric', '')} by {config.get('group_by', '')} | {config.get('time_period', '')} | {config.get('chart_type', '')} chart")
            else:
                # Empty state
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: #f8fafc; border-radius: 12px; border: 2px dashed #cbd5e1;">
                    <h3 style="color: #94a3b8; margin: 0;">📊</h3>
                    <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Configure options above and click "Generate Chart"</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Summary section at bottom
    st.markdown("---")
    st.subheader("📋 Session Summary")
    
    active_charts = sum(1 for c in st.session_state.charts.values() if c is not None)
    
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Charts Created", f"{active_charts}/3")
    with summary_cols[1]:
        st.metric("Data Records", f"{len(df):,}")
    with summary_cols[2]:
        date_range = f"{df['sale_date'].min().strftime('%b %Y')} - {df['sale_date'].max().strftime('%b %Y')}"
        st.metric("Date Range", date_range)
    with summary_cols[3]:
        ai_status = "✅ Active" if analyzer else "❌ Unavailable"
        st.metric("AI Insights", ai_status)
    
    # Clear all button
    if active_charts > 0:
        if st.button("🗑️ Clear All Charts", type="secondary"):
            st.session_state.charts = {1: None, 2: None, 3: None}
            st.session_state.chart_configs = {1: {}, 2: {}, 3: {}}
            st.session_state.ai_insights = {1: None, 2: None, 3: None}
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("💡 Charts are session-only and will reset when you close the browser. | Built for Maize Distribution Analytics")


if __name__ == "__main__":
    main()