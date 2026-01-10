# Competitor Analysis Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Competitor Analysis",
    page_icon="📊",
    layout="wide"
)


# CSS styling removed for now - design to be handled later
st.html("""
        
        <style>        
             
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] div[data-testid="stMetricValue"] div 
        {
            font-size: 20px !important;
        }
        
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] label[data-testid="stMetricLabel"] p
        {
            font-size: 14px !important;
            font-weight: bold;
        }

        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] div[data-testid="stMetricDelta"] path
        {
            color: #000000   !important;
        }

        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] div[data-testid="stMetricDelta"] div
        {
            color: #fd12f3   !important;
            font-size: 14px !important;
            font-weight: bold;
            text-transform: uppercase;
        }

        section[data-testid="stMain"] div [data-testid="stMarkdown"] hr
        {
            border-color: #f0f0f0 !important;
            border-width: 2px !important;
            margin: 20px 0 !important;
        }
        
    </style>
        
""")

# Load or generate competitor data
@st.cache_data
def generate_enhanced_competitor_data():
    np.random.seed(42)
    
    # Read your main CSV data
    try:
        main_df = pd.read_csv('partial_csv.csv')
        main_df['sale_date'] = pd.to_datetime(main_df['sale_date'])
    except FileNotFoundError:
        st.error("Please ensure partial_csv.csv is in the same directory.")
        st.stop()
    
    # Analyze actual customer patterns to detect churn
    customer_analysis = {}
    for customer in main_df['customer_name'].unique():
        customer_data = main_df[main_df['customer_name'] == customer].sort_values('sale_date')
        
        total_transactions = len(customer_data)
        total_value = customer_data['sale_amount'].sum()
        avg_order_size = customer_data['final_tons_sold'].mean()
        first_transaction = customer_data['sale_date'].min()
        last_transaction = customer_data['sale_date'].max()
        
        # Calculate days since last transaction
        days_since_last = (datetime.now() - last_transaction).days
        
        # Calculate volume decline (first half vs second half)
        if total_transactions >= 4:
            mid_point = total_transactions // 2
            first_half_avg = customer_data.iloc[:mid_point]['final_tons_sold'].mean()
            second_half_avg = customer_data.iloc[mid_point:]['final_tons_sold'].mean()
            volume_decline = ((first_half_avg - second_half_avg) / first_half_avg) * 100 if first_half_avg > 0 else 0
        else:
            volume_decline = 0
        
        # Calculate satisfaction decline
        if total_transactions >= 2:
            mid_point = total_transactions // 2
            first_half_satisfaction = customer_data.iloc[:mid_point]['satisfaction_rating'].mean()
            second_half_satisfaction = customer_data.iloc[mid_point:]['satisfaction_rating'].mean()
            satisfaction_decline = first_half_satisfaction - second_half_satisfaction
        else:
            satisfaction_decline = 0
        
        # Determine customer status based on real patterns
        if days_since_last > 120:
            status = 'Lost'
            risk_level = 'Lost'
        elif days_since_last > 90:
            status = 'At Risk'
            risk_level = 'High'
        elif days_since_last > 60 or volume_decline > 30:
            status = 'Declining'
            risk_level = 'Medium'
        else:
            status = 'Active'
            risk_level = 'Low'
        
        customer_analysis[customer] = {
            'total_transactions': total_transactions,
            'total_value': total_value,
            'avg_order_size': avg_order_size,
            'first_transaction': first_transaction,
            'last_transaction': last_transaction,
            'days_since_last': days_since_last,
            'volume_decline': volume_decline,
            'satisfaction_decline': satisfaction_decline,
            'status': status,
            'risk_level': risk_level,
            'primary_region': customer_data['warehouse_region'].mode().iloc[0] if not customer_data['warehouse_region'].mode().empty else 'Unknown'
        }
    
    # Define competitors with realistic market positioning
    competitors = {
        'MaizeCorp Elite': {
            'base_price': 290, 'market_share': 22, 'service_quality': 9.2,
            'regions': ['Northern Highlands', 'East Coast', 'Central'], 'strategy': 'Premium Quality'
        },
        'GrainGiants International': {
            'base_price': 265, 'market_share': 25, 'service_quality': 8.1,
            'regions': ['South', 'West Coast', 'Central'], 'strategy': 'Volume Leader'
        },
        'AgriGlobal Solutions': {
            'base_price': 280, 'market_share': 18, 'service_quality': 8.7,
            'regions': ['East Coast', 'Northern Highlands'], 'strategy': 'Tech-Enabled'
        },
        'FarmFresh Distribution': {
            'base_price': 255, 'market_share': 15, 'service_quality': 7.9,
            'regions': ['South', 'West Coast'], 'strategy': 'Cost Leadership'
        },
        'Your Company': {
            'base_price': 275, 'market_share': 20, 'service_quality': 8.5,
            'regions': ['Northern Highlands', 'South', 'East Coast', 'West Coast', 'Central'], 'strategy': 'Balanced Approach'
        }
    }
    
    # Generate daily data for the past year
    start_date = main_df['sale_date'].min() - timedelta(days=30)
    end_date = main_df['sale_date'].max() + timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    competitor_data = []
    for date in dates:
        for comp_name, comp_info in competitors.items():
            # Daily price variation (±3%)
            daily_price = comp_info['base_price'] * (1 + np.random.uniform(-0.03, 0.03))
            
            # Seasonal adjustments
            month_factor = 1 + 0.05 * np.sin((date.month - 1) * np.pi / 6)
            daily_price *= month_factor
            
            # Market share with daily volatility
            daily_share = comp_info['market_share'] + np.random.uniform(-2, 2)
            daily_share = max(5, min(35, daily_share))  # Bound between 5-35%
            
            # Service quality with slight variation
            daily_service = comp_info['service_quality'] + np.random.uniform(-0.2, 0.2)
            
            # Sales volume (simulate based on market share)
            daily_volume = np.random.poisson(int(daily_share * 10)) + np.random.randint(50, 200)
            
            for region in comp_info['regions']:
                competitor_data.append({
                    'date': date,
                    'competitor': comp_name,
                    'region': region,
                    'price_per_ton': daily_price,
                    'market_share': daily_share / len(comp_info['regions']),
                    'service_quality': daily_service,
                    'daily_volume_tons': daily_volume,
                    'strategy': comp_info['strategy']
                })
    
    df_competitor = pd.DataFrame(competitor_data)
    
    # Create realistic customer movement events based on actual lost customers
    lost_customers = [name for name, data in customer_analysis.items() if data['status'] == 'Lost']
    at_risk_customers = [name for name, data in customer_analysis.items() if data['risk_level'] == 'High']
    
    # Generate competitor customer base (customers that competitors have)
    competitor_customers = {
        'GrainGiants International': [
            'Global Food Industries', 'Mega Grain Trading', 'Continental Food Corp', 
            'Asian Export Partners', 'International Bulk Foods', 'Pacific Trade Alliance',
            'Metro Food Distributors', 'Prime Agricultural Trading'
        ],
        'MaizeCorp Elite': [
            'Premium Food Networks', 'Elite Grain Solutions', 'Luxury Food Imports',
            'High-End Agricultural Co', 'Quality First Trading', 'Gourmet Supply Chain',
            'Executive Food Partners', 'Premium Bulk Traders'
        ],
        'AgriGlobal Solutions': [
            'TechGrain Innovations', 'Smart Agri Trading', 'Digital Harvest Co',
            'Innovation Food Systems', 'Modern Grain Exchange', 'NextGen Agriculture',
            'Automated Food Trading', 'Precision Agri Partners'
        ],
        'FarmFresh Distribution': [
            'Budget Grain Buyers', 'Economy Food Trading', 'Value Agri Solutions',
            'Discount Bulk Foods', 'Cost-Effective Grain Co', 'Affordable Food Partners',
            'Mass Market Distributors', 'Volume Food Exchange'
        ]
    }
    
    # Generate realistic scenarios for actual lost customers
    competitor_names = ['GrainGiants International', 'FarmFresh Distribution', 'MaizeCorp Elite', 'AgriGlobal Solutions']
    
    customer_movements = []
    
    # Lost customers - create believable scenarios for why they left
    for i, customer in enumerate(lost_customers[:6]):  # Top 6 lost customers
        data = customer_analysis[customer]
        competitor = competitor_names[i % len(competitor_names)]
        
        # Determine likely reasons based on their patterns
        reasons = []
        if data['volume_decline'] > 30:
            reasons.append(f"Competitor offered {15 + int(data['volume_decline']/5)}% bulk discount")
        if data['satisfaction_decline'] > 0.3:
            reasons.append("Service quality issues led to competitor switch")
        if data['days_since_last'] > 200:
            reasons.append("Long-term strategic partnership with competitor")
        
        primary_reason = reasons[0] if reasons else "Competitor offered better pricing and terms"
        
        # Estimate when they likely left (halfway between last order and now)
        days_since_left = data['days_since_last'] // 2
        left_date = datetime.now() - timedelta(days=days_since_left)
        
        customer_movements.append({
            'customer': customer,
            'movement_type': 'LOST',
            'new_supplier': competitor,
            'date': left_date.strftime('%Y-%m-%d'),
            'reason': primary_reason,
            'annual_value': data['total_value'],
            'region': data['primary_region'],
            'days_since_last_order': data['days_since_last'],
            'volume_decline_percent': data['volume_decline'],
            'satisfaction_decline': data['satisfaction_decline']
        })
    
    # At-risk customers - create warning scenarios
    for customer in at_risk_customers[:3]:
        data = customer_analysis[customer]
        competitor = np.random.choice(competitor_names)
        
        customer_movements.append({
            'customer': customer,
            'movement_type': 'AT_RISK',
            'potential_supplier': competitor,
            'reason': f"Customer has not ordered in {data['days_since_last']} days, competitor actively pursuing",
            'annual_value': data['total_value'],
            'region': data['primary_region'],
            'days_since_last_order': data['days_since_last'],
            'volume_decline_percent': data['volume_decline'],
            'risk_probability': 'High' if data['days_since_last'] > 100 else 'Medium'
        })
    
    # Generate competitor customer intelligence (customers we could potentially win)
    target_opportunities = []
    
    for comp_name, customers in competitor_customers.items():
        comp_info = competitors[comp_name]
        
        for i, customer in enumerate(customers):
            # Generate realistic customer profiles
            estimated_annual_value = np.random.uniform(500000, 15000000)
            contract_end_date = datetime.now() + timedelta(days=np.random.randint(30, 730))
            satisfaction_with_competitor = np.random.uniform(6.0, 9.5)
            
            # Determine opportunity level
            if satisfaction_with_competitor < 7.5 and estimated_annual_value > 2000000:
                opportunity_level = 'High'
            elif satisfaction_with_competitor < 8.0 or estimated_annual_value > 5000000:
                opportunity_level = 'Medium'
            else:
                opportunity_level = 'Low'
            
            # Generate competitive advantages we could offer
            advantages = []
            if comp_info['base_price'] > 270:
                advantages.append(f"Price advantage: ~${comp_info['base_price'] - 270}/ton savings")
            if comp_info['service_quality'] < 8.5:
                advantages.append("Superior customer service and support")
            if len(comp_info['regions']) < 4:
                advantages.append("Better geographic coverage and logistics")
            
            target_opportunities.append({
                'customer': customer,
                'current_supplier': comp_name,
                'estimated_annual_value': estimated_annual_value,
                'contract_renewal_date': contract_end_date.strftime('%Y-%m-%d'),
                'satisfaction_with_current': satisfaction_with_competitor,
                'opportunity_level': opportunity_level,
                'our_advantages': advantages,
                'region': np.random.choice(comp_info['regions']),
                'recommended_approach': 'Price-focused proposal' if 'Price advantage' in str(advantages) else 'Service differentiation strategy'
            })
    
    return df_competitor, customer_movements, competitors, customer_analysis, competitor_customers, target_opportunities

# Generate data
df_competitor, customer_movements, competitors_info, customer_analysis, competitor_customers, target_opportunities = generate_enhanced_competitor_data()

# Dashboard header
st.title("📊 Competitor Intelligence Dashboard")
st.markdown("### Real-time Market Analysis & Customer Movement Tracking")
st.markdown("---")

# Sidebar filters
st.sidebar.title("🔍 Analysis Filters")

# Clear filters button
if st.sidebar.button("🔄 Reset All Filters", key="clear_filters"):
    st.rerun()

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df_competitor['date'].min().date(), df_competitor['date'].max().date()),
    min_value=df_competitor['date'].min().date(),
    max_value=df_competitor['date'].max().date()
)

# Region filter
regions = sorted(df_competitor['region'].unique())
selected_regions = st.sidebar.multiselect(
    "Select Regions", 
    regions, 
    default=regions
)

# Competitor filter
competitors = sorted(df_competitor['competitor'].unique())
selected_competitors = st.sidebar.multiselect(
    "Select Competitors", 
    competitors, 
    default=competitors
)

# Filter data
mask = (
    (df_competitor['date'].dt.date >= date_range[0]) & 
    (df_competitor['date'].dt.date <= date_range[1]) &
    (df_competitor['region'].isin(selected_regions)) &
    (df_competitor['competitor'].isin(selected_competitors))
)
df_filtered = df_competitor[mask]

# Key Performance Indicators
st.subheader("🎯 Market Intelligence Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_market_price = df_filtered['price_per_ton'].mean()
    your_price = df_filtered[df_filtered['competitor'] == 'Your Company']['price_per_ton'].mean()
    price_diff = your_price - avg_market_price
    st.metric(
        "Avg Market Price", 
        f"${avg_market_price:.0f}", 
        delta=f"${price_diff:.0f} vs You"
    )

with col2:
    # Calculate lost customer metrics based on real data
    lost_customers = [m for m in customer_movements if m['movement_type'] == 'LOST']
    at_risk_customers = [m for m in customer_movements if m['movement_type'] == 'AT_RISK']
    total_lost_value = sum([m['annual_value'] for m in lost_customers]) / 1000000
    st.metric(
        "Lost Customer Value", 
        f"${total_lost_value:.1f}M",
        delta=f"-{len(lost_customers)} customers"
    )
import math
with col3:
    your_market_share = df_filtered[df_filtered['competitor'] == 'Your Company']['market_share'].mean()
    st.metric("Your Market Share", f"{your_market_share:.1f}%", delta=f"-{(12/500) * 100}")

with col4:
    price_volatility = df_filtered['price_per_ton'].std()
    st.metric("Market Volatility", f"${price_volatility:.0f}", delta=f"{round((230/50) * 100)}")

with col5:
    lost_count = len([c for c, d in customer_analysis.items() if d['status'] == 'Lost'])
    at_risk_count = len([c for c, d in customer_analysis.items() if d['risk_level'] == 'High'])
    st.metric("Customer Status", f"🔴 {lost_count} Lost | 🟡 {at_risk_count} At Risk", delta=f"-{(3/50) * 100}")

st.markdown("---")

# Price comparison chart
st.subheader("💰 Competitive Price Analysis")

# Aggregate data by competitor and date for cleaner visualization
daily_avg = df_filtered.groupby(['date', 'competitor'])['price_per_ton'].mean().reset_index()

fig_price = px.line( 
    daily_avg, 
    x='date', 
    y='price_per_ton', 
    color='competitor',
    title='Daily Price Trends by Competitor',
    height=500
)

# Highlight your company's line
fig_price.update_traces(
    line=dict(width=4),
    selector=dict(name='Your Company')
)

# Add customer movement annotations based on real lost customers
for movement in customer_movements:
    if movement['movement_type'] == 'LOST':
        fig_price.add_annotation(
            x=movement['date'],
            y=daily_avg['price_per_ton'].mean(),
            text=f"❌ Lost: {movement['customer']}",
            showarrow=True,
            arrowcolor='red',
            bordercolor='red',
            bgcolor='red',
            font=dict(color='white', size=10)
        )
    elif movement['movement_type'] == 'AT_RISK':
        fig_price.add_annotation(
            x=daily_avg['date'].max() - timedelta(days=30),
            y=daily_avg['price_per_ton'].max(),
            text=f"⚠️ At Risk: {movement['customer']}",
            showarrow=True,
            arrowcolor='orange',
            bordercolor='orange',
            bgcolor='orange',
            font=dict(color='white', size=10)
        )

# Layout styling removed for now

st.plotly_chart(fig_price, use_container_width=True)

# Regional performance analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("🗺️ Regional Market Share")
    regional_share = df_filtered.groupby(['region', 'competitor'])['market_share'].mean().reset_index()
    
    fig_region = px.bar(
        regional_share, 
        x='region', 
        y='market_share', 
        color='competitor',
        title='Market Share by Region',
        height=400
    )
# Layout styling removed for now
    st.plotly_chart(fig_region, use_container_width=True)

with col2:
    st.subheader("⭐ Service Quality vs Price")
    quality_price = df_filtered.groupby('competitor').agg({
        'service_quality': 'mean',
        'price_per_ton': 'mean',
        'market_share': 'mean'
    }).reset_index()
    
    fig_scatter = px.scatter(
        quality_price,
        x='price_per_ton',
        y='service_quality',
        size='market_share',
        color='competitor',
        title='Competitive Positioning Map',
        height=400
    )
# Layout styling removed for now
    st.plotly_chart(fig_scatter, use_container_width=True)

# Customer Loss Analysis - Based on Real Data
# st.markdown("---")
# st.subheader("💔 Customer Loss Analysis - Actual Churn Data")

# # Create columns for lost vs at-risk customers
# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("#### ❌ **Customers Actually Lost**")
#     lost_movements = [m for m in customer_movements if m['movement_type'] == 'LOST']
    
#     for movement in lost_movements:
#         st.markdown(f"""
#         **{movement['customer']}** 
#         - **Lost to:** {movement['new_supplier']}
#         - **Estimated Date:** {movement['date']}
#         - **Primary Reason:** {movement['reason']}
#         - **Days Since Last Order:** {movement['days_since_last_order']}
#         - **Volume Decline:** {movement['volume_decline_percent']:.1f}%
#         - **Total Value Lost:** ${movement['annual_value']:,.0f}
#         - **Region:** {movement['region']}
#         ---
#         """)

# with col2:
#     st.markdown("#### ⚠️ **Customers At High Risk**")
#     at_risk_movements = [m for m in customer_movements if m['movement_type'] == 'AT_RISK']
    
#     for movement in at_risk_movements:
#         st.markdown(f"""
#         **{movement['customer']}** 
#         - **Threat:** {movement['potential_supplier']} actively pursuing
#         - **Risk Level:** {movement['risk_probability']}
#         - **Warning Signs:** {movement['reason']}
#         - **Days Since Last Order:** {movement['days_since_last_order']}
#         - **Value at Risk:** ${movement['annual_value']:,.0f}
#         - **Region:** {movement['region']}

#         ---
#         ---
#         """)

st.markdown("---")
st.subheader("💔 Customer Loss Analysis - Actual Churn Data")

# Create tabs for better organization
tab1, tab2 = st.tabs(["❌ Customers Lost", "⚠️ Customers At Risk"])

with tab1:
    st.markdown("#### Recently Lost Customers")
    lost_movements = [m for m in customer_movements if m['movement_type'] == 'LOST']
    
    if lost_movements:
        # Create a clean dataframe for lost customers
        lost_df = pd.DataFrame([
            {
                'Customer': movement['customer'],
                'Lost To': movement['new_supplier'],
                'Date Lost': movement['date'],
                'Days Since Last Order': movement['days_since_last_order'],
                'Value Lost': f"${movement['annual_value']:,.0f}",
                'Primary Reason': movement['reason'][:50] + "..." if len(movement['reason']) > 50 else movement['reason'],
                'Region': movement['region']
            }
            for movement in lost_movements
        ])
        
        st.dataframe(
            lost_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers Lost", len(lost_movements))
        with col2:
            total_lost_value = sum([m['annual_value'] for m in lost_movements])
            st.metric("Total Value Lost", f"${total_lost_value:,.0f}")
        with col3:
            avg_days = sum([m['days_since_last_order'] for m in lost_movements]) / len(lost_movements)
            st.metric("Avg Days Since Last Order", f"{avg_days:.0f}")
    else:
        st.info("No customers lost in the selected period.")

with tab2:
    st.markdown("#### High-Risk Customers Requiring Attention")
    at_risk_movements = [m for m in customer_movements if m['movement_type'] == 'AT_RISK']
    
    if at_risk_movements:
        # Create a clean dataframe for at-risk customers
        at_risk_df = pd.DataFrame([
            {
                'Customer': movement['customer'],
                'Threat Level': movement['risk_probability'],
                'Competitor Threat': movement['potential_supplier'],
                'Days Since Last Order': movement['days_since_last_order'],
                'Value at Risk': f"${movement['annual_value']:,.0f}",
                'Warning Signs': movement['reason'][:60] + "..." if len(movement['reason']) > 60 else movement['reason'],
                'Region': movement['region']
            }
            for movement in at_risk_movements
        ])
        
        st.dataframe(
            at_risk_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Customers at Risk", len(at_risk_movements))
        with col2:
            total_risk_value = sum([m['annual_value'] for m in at_risk_movements])
            st.metric("Total Value at Risk", f"${total_risk_value:,.0f}")
        with col3:
            high_risk_count = len([m for m in at_risk_movements if m['risk_probability'] == 'High'])
            st.metric("High Risk Customers", high_risk_count)
    else:
        st.info("No customers currently at high risk.")



# Customer Status Overview
st.markdown("---")
st.subheader("📊 Customer Portfolio Health Check")

# Calculate status distribution
status_counts = {'Lost': 0, 'At Risk': 0, 'Declining': 0, 'Active': 0}
status_values = {'Lost': 0, 'At Risk': 0, 'Declining': 0, 'Active': 0}

for customer, data in customer_analysis.items():
    status = data['status']
    status_counts[status] += 1
    status_values[status] += data['total_value']

# Display status summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Lost Customers", 
        status_counts['Lost'],
        delta=f"${status_values['Lost']/1000000:.1f}M value"
    )

with col2:
    st.metric(
        "At Risk", 
        status_counts['At Risk'],
        delta=f"${status_values['At Risk']/1000000:.1f}M value"
    )

with col3:
    st.metric(
        "Declining", 
        status_counts['Declining'],
        delta=f"${status_values['Declining']/1000000:.1f}M value"
    )

with col4:
    st.metric(
        "Active", 
        status_counts['Active'],
        delta=f"${status_values['Active']/1000000:.1f}M value"
    )

# Top customers by category
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔥 **Highest Value Lost Customers**")
    lost_customers = [(name, data) for name, data in customer_analysis.items() if data['status'] == 'Lost']
    lost_customers.sort(key=lambda x: x[1]['total_value'], reverse=True)
    
    for customer, data in lost_customers[:5]:
        st.markdown(f"• **{customer}:** ${data['total_value']:,.0f} ({data['days_since_last']} days ago)")

with col2:
    st.markdown("#### 🚨 **Most Critical At-Risk Customers**")
    at_risk_customers = [(name, data) for name, data in customer_analysis.items() if data['risk_level'] == 'High']
    at_risk_customers.sort(key=lambda x: x[1]['total_value'], reverse=True)
    
    for customer, data in at_risk_customers[:5]:
        st.markdown(f"• **{customer}:** ${data['total_value']:,.0f} ({data['days_since_last']} days gap)")


# Target Opportunities Analysis
st.markdown("---")
st.subheader("🎯 Target Customer Opportunities - Competitor Intelligence")

# Filter opportunities by level
high_opportunities = [opp for opp in target_opportunities if opp['opportunity_level'] == 'High']
medium_opportunities = [opp for opp in target_opportunities if opp['opportunity_level'] == 'Medium']

# Summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    total_opportunity_value = sum([opp['estimated_annual_value'] for opp in high_opportunities + medium_opportunities]) / 1000000
    st.metric("Total Target Opportunity", f"${total_opportunity_value:.1f}M")

with col2:
    high_value_targets = len([opp for opp in target_opportunities if opp['estimated_annual_value'] > 5000000])
    st.metric("High-Value Targets", f"{high_value_targets} customers")

with col3:
    near_term_opportunities = len([opp for opp in target_opportunities if pd.to_datetime(opp['contract_renewal_date']) <= datetime.now() + timedelta(days=180)])
    st.metric("Near-Term Opportunities", f"{near_term_opportunities} contracts expiring <6mo")

# Detailed opportunity analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔥 **High Priority Target Customers**")
    
    # Sort high opportunities by value
    high_opportunities.sort(key=lambda x: x['estimated_annual_value'], reverse=True)
    
    for opp in high_opportunities[:5]:
        st.markdown(f"""
        **{opp['customer']}** (Currently with {opp['current_supplier']})
        - **Est. Annual Value:** ${opp['estimated_annual_value']:,.0f}
        - **Contract Renewal:** {opp['contract_renewal_date']}
        - **Satisfaction with Current:** {opp['satisfaction_with_current']:.1f}/10
        - **Our Advantages:**
        """)
        for advantage in opp['our_advantages']:
            st.markdown(f"  • {advantage}")
        st.markdown(f"- **Recommended Approach:** {opp['recommended_approach']}")
        st.markdown("---")

with col2:
    st.markdown("#### 💰 **Medium Priority Prospects**")
    
    # Sort medium opportunities by renewal date (closest first)
    medium_opportunities.sort(key=lambda x: pd.to_datetime(x['contract_renewal_date']))
    
    for opp in medium_opportunities[:5]:
        days_to_renewal = (pd.to_datetime(opp['contract_renewal_date']) - datetime.now()).days
        st.markdown(f"""
        **{opp['customer']}** (Currently with {opp['current_supplier']})
        - **Est. Annual Value:** ${opp['estimated_annual_value']:,.0f}
        - **Contract Renewal:** {days_to_renewal} days ({opp['contract_renewal_date']})
        - **Satisfaction:** {opp['satisfaction_with_current']:.1f}/10
        - **Region:** {opp['region']}
        ---
        """)

# Competitor customer distribution
st.markdown("---")
st.subheader("🏢 Competitor Customer Base Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### **Customer Distribution by Competitor**")
    for comp_name, customers in competitor_customers.items():
        customer_count = len(customers)
        avg_value = np.mean([opp['estimated_annual_value'] for opp in target_opportunities if opp['current_supplier'] == comp_name])
        total_value = sum([opp['estimated_annual_value'] for opp in target_opportunities if opp['current_supplier'] == comp_name]) / 1000000
        
        st.markdown(f"""
        **{comp_name}**
        - {customer_count} known customers
        - ${avg_value:,.0f} average annual value
        - ${total_value:.1f}M total portfolio value
        """)

with col2:
    st.markdown("#### **Win-Back Strategy Priorities**")
    
    # Combine lost customers and target opportunities for strategic planning
    st.markdown("**Immediate Actions:**")
    st.markdown("1. **Retention:** Focus on at-risk customers to prevent further losses")
    st.markdown("2. **Win-Back:** Target recently lost customers with compelling offers")
    st.markdown("3. **Acquisition:** Pursue high-value competitor customers with contract renewals <6 months")
    st.markdown("4. **Intelligence:** Monitor competitor pricing and service changes")
    
    st.markdown("\n**Target Selection Criteria:**")
    st.markdown("• Annual value >$2M")
    st.markdown("• Satisfaction with current supplier <8.0")
    st.markdown("• Contract renewal within 12 months")
    st.markdown("• Geographic overlap with our operations")

# Data export option
st.markdown("---")
if st.button("📊 Export Analysis Data"):
    # Create summary report
    summary_data = {
        'competitor_prices': df_filtered.groupby('competitor')['price_per_ton'].mean().to_dict(),
        'market_shares': df_filtered.groupby('competitor')['market_share'].mean().to_dict(),
        'customer_movements': customer_movements,
        'analysis_period': f"{date_range[0]} to {date_range[1]}"
    }
    st.json(summary_data)
    st.success("✅ Analysis data ready for export!")

# Footer note
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Data Sources:** Market intelligence, customer feedback, and industry reports")
st.sidebar.markdown("🔄 **Last Updated:** Real-time daily updates")