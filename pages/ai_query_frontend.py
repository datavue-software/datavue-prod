"""
AI Query Assistant - Frontend Page
Ask questions in plain English, get SQL results
Fixed: Removed debug output, fixed button reset issue
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# Import the query assistant
import ai_query_assistant


# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="AI Query Assistant | Maize Analytics",
    page_icon="🤖",
    layout="wide"
)


# ====== CACHED RESOURCES ======
@st.cache_resource
def get_query_assistant():
    """Get cached query assistant instance"""
    return ai_query_assistant.get_cached_assistant()


# ====== STYLING ======
def apply_page_styling():
    """Apply CSS styling for AI Query page"""
    st.markdown("""
    <style>
    /* Main container styling */
    .ai-query-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        color: white;
    }
    
    .ai-query-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .ai-query-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .ai-query-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    /* Stats cards */
    .stats-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
    }
    
    .stats-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .stats-card h2 {
        margin: 0;
        font-size: 1.5rem;
    }
    
    /* SQL display */
    .sql-display {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ====== EXAMPLE PROMPTS ======
EXAMPLE_PROMPTS = [
    "💰 What are our total sales this year?",
    "🏆 Show me top 10 customers by revenue",
    "📊 Which products sell best?",
    "🌍 Sales breakdown by region",
    "📈 Monthly revenue trends",
    "⭐ Customers with high satisfaction",
    "🎯 Export vs Local sales comparison",
    "📦 Average order size by category",
    "💎 Premium customers (>$100k revenue)",
    "🔍 Sales in Northern region"
]


# ====== DISPLAY FUNCTIONS ======
def display_results(result_dict, question):
    """Display query results with stats and download options"""
    if result_dict.get('error'):
        st.error(f"❌ **Query Failed:** {result_dict['error']}")
        st.info("💡 **Tip:** Try rephrasing your question or check for typos")
        return
    
    df_result = result_dict['result']
    sql_query = result_dict['sql']
    
    if df_result is None or len(df_result) == 0:
        st.warning("🔍 No results found for your query. Try a different question!")
        return
    
    # Success header
    st.success(f"✅ **Query Successful!** Found {len(df_result)} results")
    
    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <h4>📊 Rows</h4>
            <h2>{len(df_result)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <h4>📋 Columns</h4>
            <h2>{len(df_result.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Check for sale_amount column
        if 'sale_amount' in df_result.columns:
            total_value = df_result['sale_amount'].sum()
            st.markdown(f"""
            <div class="stats-card">
                <h4>💰 Total Value</h4>
                <h2>${total_value:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stats-card">
                <h4>🔢 Data Type</h4>
                <h2>Mixed</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-card">
            <h4>⚡ Status</h4>
            <h2>Ready</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Action buttons row
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        # Download CSV
        csv = df_result.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with action_col2:
        # Copy SQL button (shows in expander)
        if st.button("📋 Copy SQL", use_container_width=True, key="copy_sql_btn"):
            st.session_state.show_sql = True
    
    with action_col3:
        # Explain Query
        if st.button("📖 Explain Query", use_container_width=True, key="explain_btn"):
            st.session_state.show_explanation = True
    
    with action_col4:
        # Toggle chart section
        if st.button("📊 Create Chart", use_container_width=True, key="chart_toggle_btn"):
            st.session_state.show_chart_builder = not st.session_state.get('show_chart_builder', False)
    
    # SQL Display (shown when requested)
    if st.session_state.get('show_sql', False):
        with st.expander("🔧 Generated SQL Query", expanded=True):
            st.code(sql_query, language="sql")
            st.caption("Copy the SQL above to use elsewhere")
            if st.button("Close", key="close_sql"):
                st.session_state.show_sql = False
                st.rerun()
    
    # Query Explanation (shown when requested)
    if st.session_state.get('show_explanation', False):
        with st.expander("📖 Query Explanation", expanded=True):
            st.markdown(f"""
            **Your Question:** {question}
            
            **Generated SQL:** 
            ```sql
            {sql_query}
            ```
            
            **What it does:** This query searches the sales database based on your question 
            and returns matching records. The AI interpreted your natural language and 
            converted it to SQL that the database can understand.
            """)
            if st.button("Close", key="close_explain"):
                st.session_state.show_explanation = False
                st.rerun()
    
    # Results Table
    st.markdown("### 📋 Query Results")
    st.dataframe(
        df_result.style.format(precision=2),
        use_container_width=True,
        height=min(400, len(df_result) * 35 + 100)
    )
    
    # Chart Builder Section
    if st.session_state.get('show_chart_builder', False):
        display_chart_builder(df_result)


def display_chart_builder(df):
    """Display chart builder section"""
    st.markdown("---")
    st.markdown("### 📊 Quick Chart Builder")
    
    # Get column info
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    all_cols = list(df.columns)
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns available for charting")
        return
    
    # Chart configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Pie Chart", "Line Chart", "Scatter Plot"],
            key="chart_type_select"
        )
    
    with col2:
        x_column = st.selectbox(
            "X-Axis / Category",
            all_cols,
            key="x_col_select"
        )
    
    with col3:
        y_column = st.selectbox(
            "Y-Axis / Value",
            numeric_cols,
            key="y_col_select"
        )
    
    # Generate chart button
    if st.button("📈 Generate Chart", type="primary", key="gen_chart_btn"):
        try:
            fig = None
            title = f"{y_column} by {x_column}"
            
            if chart_type == "Bar Chart":
                chart_data = df.groupby(x_column)[y_column].sum().reset_index()
                fig = px.bar(chart_data, x=x_column, y=y_column, title=title)
            
            elif chart_type == "Pie Chart":
                chart_data = df.groupby(x_column)[y_column].sum().reset_index()
                if len(chart_data) > 10:
                    chart_data = chart_data.nlargest(10, y_column)
                fig = px.pie(chart_data, values=y_column, names=x_column, title=title)
            
            elif chart_type == "Line Chart":
                chart_data = df.groupby(x_column)[y_column].sum().reset_index()
                chart_data = chart_data.sort_values(x_column)
                fig = px.line(chart_data, x=x_column, y=y_column, title=title, markers=True)
            
            else:  # Scatter
                fig = px.scatter(df, x=x_column, y=y_column, title=title)
            
            if fig:
                fig.update_layout(
                    plot_bgcolor='rgba(248, 250, 252, 0.8)',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI", size=12),
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Chart creation failed: {e}")
            st.info("Try different column combinations")


# ====== MAIN PAGE ======
def main():
    # Apply styling
    apply_page_styling()
    
    # Initialize session state
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'show_sql' not in st.session_state:
        st.session_state.show_sql = False
    if 'show_explanation' not in st.session_state:
        st.session_state.show_explanation = False
    if 'show_chart_builder' not in st.session_state:
        st.session_state.show_chart_builder = False
    
    # Header
    st.markdown("""
    <div class="ai-query-container">
        <div class="ai-query-header">
            <h1 class="ai-query-title">🤖 AI Query Assistant</h1>
            <p style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 1rem; font-style: italic;">
                Compiled by <strong>Datavue</strong> • Powered by AI
            </p>
            <p class="ai-query-subtitle">Ask anything about your sales data in plain English</p>
            <p style="opacity: 0.8; font-size: 1rem;">No SQL knowledge required • Instant insights</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Example prompts section
    st.markdown("### 💡 Try These Example Questions")
    
    # Display examples in grid
    cols = st.columns(5)
    for i, example in enumerate(EXAMPLE_PROMPTS):
        with cols[i % 5]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                # Extract text without emoji
                st.session_state.selected_example = example.split(' ', 1)[1]
                st.rerun()
    
    st.markdown("---")
    
    # Main query input
    st.markdown("### 🎯 Ask Your Question")
    
    # Get default value from selected example if exists
    default_value = ""
    if 'selected_example' in st.session_state:
        default_value = st.session_state.selected_example
        del st.session_state.selected_example
    
    # Query input
    user_question = st.text_input(
        "Type your question here:",
        value=default_value,
        placeholder="e.g., What are the top 5 customers by revenue this year?",
        key="query_input",
        label_visibility="collapsed"
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_clicked = st.button(
            "🚀 Ask AI",
            type="primary",
            use_container_width=True,
            key="search_btn"
        )
    
    with col2:
        if st.button("🎲 Random", use_container_width=True, key="random_btn"):
            random_example = random.choice(EXAMPLE_PROMPTS)
            st.session_state.selected_example = random_example.split(' ', 1)[1]
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
            st.session_state.current_result = None
            st.session_state.current_question = None
            st.session_state.show_sql = False
            st.session_state.show_explanation = False
            st.session_state.show_chart_builder = False
            st.rerun()
    
    # Execute query when search is clicked
    if search_clicked and user_question.strip():
        with st.spinner("🤖 Processing your query..."):
            # Get cached assistant
            assistant = get_query_assistant()
            
            # Execute query
            result = assistant.query(user_question.strip())
            
            # Store in session state
            st.session_state.current_result = result
            st.session_state.current_question = user_question.strip()
            st.session_state.show_chart_builder = False  # Reset chart builder
            
            st.rerun()
    
    elif search_clicked and not user_question.strip():
        st.warning("⚠️ Please enter a question before searching!")
    
    # Display results if we have them (THIS IS THE KEY FIX - display outside button check)
    if st.session_state.current_result is not None:
        st.markdown("---")
        display_results(
            st.session_state.current_result,
            st.session_state.current_question
        )
    
    # Query history in sidebar
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Add to history on successful query
    if search_clicked and user_question.strip():
        history_entry = {
            'question': user_question.strip(),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        # Avoid duplicates
        if not any(h['question'] == history_entry['question'] for h in st.session_state.query_history):
            st.session_state.query_history.append(history_entry)
            # Keep only last 10
            st.session_state.query_history = st.session_state.query_history[-10:]
    
    # Show history
    if st.session_state.query_history:
        st.markdown("---")
        with st.expander("📝 Recent Queries", expanded=False):
            for i, query in enumerate(reversed(st.session_state.query_history[-5:])):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"🕐 {query['timestamp']} - {query['question'][:60]}{'...' if len(query['question']) > 60 else ''}")
                with col2:
                    if st.button("↩️", key=f"rerun_{i}", help="Run this query again"):
                        st.session_state.selected_example = query['question']
                        st.rerun()


if __name__ == "__main__":
    main()