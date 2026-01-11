# Plugins Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Business Plugins",
    page_icon="🔌",
    layout="wide"
)

# Apply consistent styling from your main app
def apply_styling():
    st.html("""
        <style>
            /* Sidebar styling */
            section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
                color: #ffffff;
            }

            section[data-testid="stSidebar"] {
                background: #3f3c4d;
                border: 2px solid #36353a;
                border-top-left-radius: 0px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                border-top-right-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            }

            section[data-testid="stSidebar"] h1 {
                color: #ffffff;
                font-size: 1.7rem;
            }

            section[data-testid="stSidebar"] p {
                color: #ffffff;
            }

            /* Hide header */
            header[data-testid="stHeader"] {
                background: transparent;
                visibility: hidden;
                height: 0px;
                color: white !important;
            }

            /* Main app background */
            .stApp {
                background: linear-gradient(135deg, #66e6ff 0%, #418FDE 100%);
            }

            /* Plugin card styling */
            .plugin-card {
                background: rgba(255, 255, 255, 0.95);
                border: 2px solid #5a67d8;
                border-radius: 16px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                transition: all 0.3s ease;
            }

            .plugin-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2);
            }

            .plugin-header {
                display: flex;
                align-items: center;
                margin-bottom: 1rem;
            }

            .plugin-icon {
                font-size: 2rem;
                margin-right: 1rem;
            }

            .plugin-title {
                font-size: 1.5rem;
                font-weight: bold;
                color: #1F2B3A;
                margin: 0;
            }

            .plugin-description {
                color: #666;
                margin-bottom: 1rem;
            }

            .plugin-status {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 500;
            }

            .status-active {
                background: #d4edda;
                color: #155724;
            }

            .status-inactive {
                background: #f8d7da;
                color: #721c24;
            }

            /* Metrics styling consistent with main app */
            div[data-testid="stMetric"] {
                background: linear-gradient(135deg, #66e6ff 0%, #418FDE 100%);
                border: 2px solid #5a67d8;
                border-radius: 16px;
                padding: 2rem 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.5rem;
                min-height: 150px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            div[data-testid="stMetric"] * {
                color: white !important;
            }

            /* Button styling for plugin management */
            .stButton button {
                border-radius: 8px;
                border: none;
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.2s ease;
            }

            /* Chart styling */
            div[data-testid="stPlotlyChart"] > div {
                border-radius: 16px !important;
                overflow: hidden;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.20);
                border: 2px solid rgba(65, 143, 222, 0.2);
            }

            div[data-testid="stPlotlyChart"] svg {
                border-radius: 14px;
            }
        </style>
    """)

# apply_styling()

# Plugin definitions with dummy data generators
AVAILABLE_PLUGINS = {
    "weather": {
        "name": "Weather Dashboard",
        "icon": "🌤️",
        "description": "Real-time weather data and forecasts for your business locations",
        "category": "Operations",
        "data_generator": lambda: {
            "current_temp": np.random.randint(65, 85),
            "humidity": np.random.randint(40, 80),
            "wind_speed": np.random.randint(5, 15),
            "forecast": [
                {"day": "Today", "high": 82, "low": 68, "condition": "Sunny"},
                {"day": "Tomorrow", "high": 79, "low": 65, "condition": "Partly Cloudy"},
                {"day": "Wednesday", "high": 75, "low": 62, "condition": "Rainy"},
                {"day": "Thursday", "high": 81, "low": 67, "condition": "Sunny"},
                {"day": "Friday", "high": 83, "low": 70, "condition": "Clear"}
            ]
        }
    },
    "finance": {
        "name": "Financial Markets",
        "icon": "💹",
        "description": "Stock market data, commodity prices, and financial indicators",
        "category": "Finance",
        "data_generator": lambda: {
            "market_indices": {
                "S&P 500": {"value": 4567.89, "change": 23.45, "change_pct": 0.52},
                "NASDAQ": {"value": 14567.23, "change": -45.67, "change_pct": -0.31},
                "DOW": {"value": 35123.45, "change": 156.78, "change_pct": 0.45}
            },
            "commodities": {
                "Corn": {"price": 6.85, "change": 0.12, "unit": "$/bushel"},
                "Wheat": {"price": 8.92, "change": -0.05, "unit": "$/bushel"},
                "Soybeans": {"price": 14.23, "change": 0.34, "unit": "$/bushel"}
            }
        }
    },
    "economy": {
        "name": "Economic Indicators",
        "icon": "📊",
        "description": "Key economic data including inflation, employment, and GDP metrics",
        "category": "Analytics",
        "data_generator": lambda: {
            "indicators": {
                "Unemployment Rate": {"value": 3.7, "change": -0.1, "unit": "%"},
                "Inflation Rate": {"value": 3.2, "change": 0.2, "unit": "%"},
                "GDP Growth": {"value": 2.4, "change": 0.1, "unit": "%"},
                "Interest Rate": {"value": 5.25, "change": 0.0, "unit": "%"}
            },
            "consumer_confidence": 104.2,
            "retail_sales_growth": 2.8
        }
    },
    "quickbooks": {
        "name": "QuickBooks Integration",
        "icon": "💼",
        "description": "High-level overview of your QuickBooks accounts and financial health",
        "category": "Accounting",
        "data_generator": lambda: {
            "accounts": {
                "Checking Account": {"balance": 245678.90, "change": 12345.67},
                "Savings Account": {"balance": 89234.56, "change": 2345.67},
                "Accounts Receivable": {"balance": 156789.12, "change": -5678.90},
                "Accounts Payable": {"balance": 78945.23, "change": 3456.78}
            },
            "recent_transactions": [
                {"date": "2025-06-14", "description": "Payment from Global Foods", "amount": 15678.90},
                {"date": "2025-06-13", "description": "Office Supplies", "amount": -456.78},
                {"date": "2025-06-12", "description": "Warehouse Rental", "amount": -8900.00},
                {"date": "2025-06-11", "description": "Customer Payment", "amount": 23456.78}
            ],
            "profit_loss": {
                "revenue": 2456789.12,
                "expenses": 1876543.45,
                "net_income": 580245.67
            }
        }
    },
    "social_media": {
        "name": "Social Media Analytics",
        "icon": "📱",
        "description": "Track social media mentions, engagement, and brand sentiment",
        "category": "Marketing",
        "data_generator": lambda: {
            "platforms": {
                "LinkedIn": {"followers": 12450, "engagement_rate": 4.2, "mentions": 45},
                "Twitter": {"followers": 8970, "engagement_rate": 2.8, "mentions": 23},
                "Facebook": {"followers": 15630, "engagement_rate": 3.5, "mentions": 67}
            },
            "sentiment": {"positive": 72, "neutral": 18, "negative": 10},
            "trending_topics": ["sustainable agriculture", "grain prices", "harvest season"]
        }
    },
    "logistics": {
        "name": "Logistics Tracker",
        "icon": "🚛",
        "description": "Real-time shipping costs, delivery tracking, and supply chain metrics",
        "category": "Operations",
        "data_generator": lambda: {
            "active_shipments": 23,
            "on_time_delivery": 94.5,
            "avg_shipping_cost": 145.67,
            "routes": [
                {"from": "Chicago", "to": "Denver", "status": "In Transit", "eta": "2025-06-16"},
                {"from": "Kansas City", "to": "Phoenix", "status": "Delivered", "eta": "2025-06-15"},
                {"from": "Des Moines", "to": "Seattle", "status": "Loading", "eta": "2025-06-18"}
            ]
        }
    }
}

# Initialize session state for active plugins
if 'active_plugins' not in st.session_state:
    st.session_state.active_plugins = []

# Dashboard header
st.title("🔌 Business Intelligence Plugins")
st.subheader("Extend your dashboard with external data sources")
st.markdown("---")

# Plugin management section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📦 Available Plugins")
    
    # Create plugin cards
    for plugin_id, plugin_info in AVAILABLE_PLUGINS.items():
        is_active = plugin_id in st.session_state.active_plugins
        
        with st.container():
            st.markdown(f"""
                <div class="plugin-card">
                    <div class="plugin-header">
                        <span class="plugin-icon">{plugin_info['icon']}</span>
                        <h3 class="plugin-title">{plugin_info['name']}</h3>
                    </div>
                    <p class="plugin-description">{plugin_info['description']}</p>
                    <span class="plugin-status {'status-active' if is_active else 'status-inactive'}">
                        {'🟢 Active' if is_active else '🔴 Inactive'}
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            # Plugin action buttons
            button_col1, button_col2, button_col3 = st.columns([1, 1, 2])
            
            with button_col1:
                if not is_active:
                    if st.button(f"Add {plugin_info['name']}", key=f"add_{plugin_id}"):
                        st.session_state.active_plugins.append(plugin_id)
                        st.success(f"✅ {plugin_info['name']} added successfully!")
                        st.rerun()
                else:
                    if st.button(f"Remove {plugin_info['name']}", key=f"remove_{plugin_id}"):
                        st.session_state.active_plugins.remove(plugin_id)
                        st.success(f"❌ {plugin_info['name']} removed successfully!")
                        st.rerun()
            
            with button_col2:
                if is_active:
                    if st.button(f"Configure", key=f"config_{plugin_id}"):
                        st.info(f"🔧 Configuration for {plugin_info['name']} (Feature coming soon)")

with col2:
    st.subheader("⚙️ Plugin Manager")
    
    # Display active plugins summary
    st.metric("Active Plugins", len(st.session_state.active_plugins))
    
    if st.session_state.active_plugins:
        st.markdown("**Currently Active:**")
        for plugin_id in st.session_state.active_plugins:
            plugin = AVAILABLE_PLUGINS[plugin_id]
            st.markdown(f"• {plugin['icon']} {plugin['name']}")
    else:
        st.info("No plugins currently active. Add plugins from the left panel to get started!")
    
    # Quick actions
    st.markdown("---")
    st.markdown("**Quick Actions:**")
    
    if st.button("🔄 Refresh All Data"):
        st.success("All plugin data refreshed!")
        st.rerun()
    
    if st.button("❌ Remove All Plugins"):
        if st.session_state.active_plugins:
            st.session_state.active_plugins = []
            st.success("All plugins removed!")
            st.rerun()
        else:
            st.warning("No active plugins to remove.")

# Display active plugin data
if st.session_state.active_plugins:
    st.markdown("---")
    st.subheader("📊 Active Plugin Data")
    
    # Create tabs for each active plugin
    tabs = st.tabs([f"{AVAILABLE_PLUGINS[pid]['icon']} {AVAILABLE_PLUGINS[pid]['name']}" for pid in st.session_state.active_plugins])
    
    for i, plugin_id in enumerate(st.session_state.active_plugins):
        plugin_info = AVAILABLE_PLUGINS[plugin_id]
        
        with tabs[i]:
            # Generate and display plugin data
            data = plugin_info['data_generator']()
            
            if plugin_id == "weather":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temperature", f"{data['current_temp']}°F")
                with col2:
                    st.metric("Humidity", f"{data['humidity']}%")
                with col3:
                    st.metric("Wind Speed", f"{data['wind_speed']} mph")
                
                # 5-day forecast
                st.subheader("5-Day Forecast")
                forecast_data = pd.DataFrame(data['forecast'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast_data['day'], y=forecast_data['high'], 
                                       mode='lines+markers', name='High', line=dict(color='#FF6B35')))
                fig.add_trace(go.Scatter(x=forecast_data['day'], y=forecast_data['low'], 
                                       mode='lines+markers', name='Low', line=dict(color='#418FDE')))
                fig.update_layout(title="Temperature Forecast", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plugin_id == "finance":
                st.subheader("Market Indices")
                col1, col2, col3 = st.columns(3)
                
                for i, (index, info) in enumerate(data['market_indices'].items()):
                    col = [col1, col2, col3][i]
                    with col:
                        delta_color = "normal" if info['change'] >= 0 else "inverse"
                        st.metric(index, f"{info['value']:,.2f}", 
                                f"{info['change']:+.2f} ({info['change_pct']:+.2f}%)", 
                                delta_color=delta_color)
                
                st.subheader("Commodity Prices")
                commodity_df = pd.DataFrame(data['commodities']).T
                commodity_df['price_display'] = commodity_df.apply(lambda x: f"${x['price']:.2f} {x['unit']}", axis=1)
                commodity_df['change_display'] = commodity_df['change'].apply(lambda x: f"{x:+.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(commodity_df[['price_display', 'change_display']], 
                               column_config={
                                   'price_display': 'Current Price',
                                   'change_display': 'Daily Change'
                               })
            
            elif plugin_id == "economy":
                st.subheader("Key Economic Indicators")
                
                # Display indicators in a grid
                cols = st.columns(2)
                for i, (indicator, info) in enumerate(data['indicators'].items()):
                    col = cols[i % 2]
                    with col:
                        delta_color = "normal" if info['change'] >= 0 else "inverse"
                        st.metric(indicator, f"{info['value']}{info['unit']}", 
                                f"{info['change']:+.1f}", delta_color=delta_color)
                
                # Additional metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Consumer Confidence", f"{data['consumer_confidence']:.1f}")
                with col2:
                    st.metric("Retail Sales Growth", f"{data['retail_sales_growth']:.1f}%")
            
            elif plugin_id == "quickbooks":
                st.subheader("Account Balances")
                
                # Account balances
                col1, col2 = st.columns(2)
                for i, (account, info) in enumerate(data['accounts'].items()):
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        delta_color = "normal" if info['change'] >= 0 else "inverse"
                        st.metric(account, f"${info['balance']:,.2f}", 
                                f"${info['change']:+,.2f}", delta_color=delta_color)
                
                # Profit & Loss Summary
                st.subheader("Profit & Loss Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Revenue", f"${data['profit_loss']['revenue']:,.2f}")
                with col2:
                    st.metric("Expenses", f"${data['profit_loss']['expenses']:,.2f}")
                with col3:
                    st.metric("Net Income", f"${data['profit_loss']['net_income']:,.2f}")
                
                # Recent transactions
                st.subheader("Recent Transactions")
                transactions_df = pd.DataFrame(data['recent_transactions'])
                transactions_df['amount'] = transactions_df['amount'].apply(lambda x: f"${x:,.2f}")
                st.dataframe(transactions_df, use_container_width=True)
            
            elif plugin_id == "social_media":
                st.subheader("Social Media Overview")
                
                # Platform metrics
                for platform, metrics in data['platforms'].items():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{platform} Followers", f"{metrics['followers']:,}")
                    with col2:
                        st.metric(f"{platform} Engagement", f"{metrics['engagement_rate']:.1f}%")
                    with col3:
                        st.metric(f"{platform} Mentions", metrics['mentions'])
                
                # Sentiment analysis
                st.subheader("Brand Sentiment")
                sentiment_df = pd.DataFrame([data['sentiment']])
                fig = px.pie(values=list(data['sentiment'].values()), 
                           names=list(data['sentiment'].keys()),
                           title="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            elif plugin_id == "logistics":
                st.subheader("Logistics Overview")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Active Shipments", data['active_shipments'])
                with col2:
                    st.metric("On-Time Delivery", f"{data['on_time_delivery']:.1f}%")
                with col3:
                    st.metric("Avg Shipping Cost", f"${data['avg_shipping_cost']:.2f}")
                
                # Active routes
                st.subheader("Active Routes")
                routes_df = pd.DataFrame(data['routes'])
                st.dataframe(routes_df, use_container_width=True)

else:
    # Show empty state with suggestions
    st.markdown("---")
    st.info("👆 **Get Started:** Add plugins from the available options above to see your business data come to life!")
    
    # Show plugin categories
    st.subheader("📂 Plugin Categories")
    
    categories = {}
    for plugin_id, plugin_info in AVAILABLE_PLUGINS.items():
        category = plugin_info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(plugin_info)
    
    cols = st.columns(len(categories))
    for i, (category, plugins) in enumerate(categories.items()):
        with cols[i]:
            st.markdown(f"**{category}**")
            for plugin in plugins:
                st.markdown(f"• {plugin['icon']} {plugin['name']}")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Plugin Manager")
st.sidebar.markdown(f"**Active Plugins:** {len(st.session_state.active_plugins)}")
st.sidebar.markdown(f"**Available Plugins:** {len(AVAILABLE_PLUGINS)}")

if st.session_state.active_plugins:
    st.sidebar.markdown("**Quick Remove:**")
    for plugin_id in st.session_state.active_plugins:
        plugin = AVAILABLE_PLUGINS[plugin_id]
        if st.sidebar.button(f"❌ {plugin['name']}", key=f"sidebar_remove_{plugin_id}"):
            st.session_state.active_plugins.remove(plugin_id)
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Note:** Plugin data is simulated for demonstration. Real integrations will connect to actual APIs and services.")
st.sidebar.markdown("🔄 **Auto-refresh:** Every 5 minutes when active")