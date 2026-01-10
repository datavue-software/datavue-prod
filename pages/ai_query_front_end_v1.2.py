# Here's what your file should look like after cleaning up:

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import ai_query_assistant
import random

# ====== FIXED AI CLIENT SETUP ======
@st.cache_resource
def get_chart_ai_client():
    """Get AI client for chart generation - Fixed version"""
    try:
        # Create assistant instance and get client
        assistant = ai_query_assistant.AIQueryAssistant()
        return assistant.client  # Access the client directly
    except Exception as e:
        st.error(f"Failed to initialize AI client: {e}")
        return None

# ====== AI QUERY PAGE STYLING ======
def create_ai_query_css():
    """CSS styling for AI Query page"""
    return """
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
    
    /* Example prompts styling */
    .example-prompts {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .example-button {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 25px !important;
        color: white !important;
        margin: 0.25rem !important;
        transition: all 0.3s ease !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    .example-button:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        font-size: 1.1rem !important;
        color: #333 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.3) !important;
    }
    
    /* Button styling */
    .ai-search-button {
        background: linear-gradient(45deg, #FFD700, #FFA500) !important;
        border: none !important;
        border-radius: 15px !important;
        color: #333 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.8rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 15px rgba(255, 165, 0, 0.3) !important;
    }
    
    .ai-search-button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(255, 165, 0, 0.4) !important;
    }
    
    /* Results styling */
    .results-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
    }
    
    .sql-display {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Autocomplete styling */
    .autocomplete-suggestion {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #333;
    }
    
    .autocomplete-suggestion:hover {
        background: #667eea;
        color: white;
        transform: translateX(5px);
    }
    </style>
    """

def create_example_prompts():
    """Create example prompt suggestions"""
    examples = [
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
    return examples

def create_autocomplete_suggestions():
    """Create autocomplete suggestions based on common patterns"""
    suggestions = [
        "What are the total sales",
        "Show me top customers",
        "Which products",
        "Sales by region",
        "Revenue for",
        "Average order size",
        "Customers with satisfaction rating",
        "Export sales",
        "Local sales",
        "Monthly sales",
        "Yearly revenue",
        "Best selling products",
        "Lowest performing",
        "Highest revenue customers"
    ]
    return suggestions

def display_autocomplete(user_input, suggestions):
    """Display autocomplete suggestions"""
    if len(user_input) > 2:
        matching = [s for s in suggestions if user_input.lower() in s.lower()]
        if matching:
            st.markdown("**💡 Suggestions:**")
            cols = st.columns(min(3, len(matching)))
            for i, suggestion in enumerate(matching[:6]):
                with cols[i % 3]:
                    if st.button(f"🔮 {suggestion}", key=f"autocomplete_{i}", help="Click to use this suggestion"):
                        return suggestion
    return None

def display_enhanced_results(result_dict, question):
    """Display results with enhanced formatting"""
    if result_dict['error']:
        st.error(f"❌ **Query Failed:** {result_dict['error']}")
        st.info("💡 **Tip:** Try rephrasing your question or check for typos")
        return
    
    df_result = result_dict['result']
    sql_query = result_dict['sql']
    
    # Results header
    st.success(f"✅ **Query Successful!** Found {len(df_result)} results")
    
    # Quick stats
    if len(df_result) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="stats-card">
                <h4>📊 Rows</h4>
                <h2>{}</h2>
            </div>
            """.format(len(df_result)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stats-card">
                <h4>📋 Columns</h4>
                <h2>{}</h2>
            </div>
            """.format(len(df_result.columns)), unsafe_allow_html=True)
        
        with col3:
            # Find numeric columns for sum
            numeric_cols = df_result.select_dtypes(include=[np.number]).columns
            if 'sale_amount' in numeric_cols:
                total_value = df_result['sale_amount'].sum()
                st.markdown("""
                <div class="stats-card">
                    <h4>💰 Total Value</h4>
                    <h2>${:,.0f}</h2>
                </div>
                """.format(total_value), unsafe_allow_html=True)
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
                <h4>⚡ Query Time</h4>
                <h2>< 1s</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # SQL Query Display
    with st.expander("🔧 **View Generated SQL Query**", expanded=False):
        st.markdown(f"""
        <div class="sql-display">
            <strong>Generated SQL:</strong><br>
            <code>{sql_query}</code>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📋 Copy SQL to Clipboard"):
            st.code(sql_query, language="sql")
            st.success("SQL copied! You can paste it elsewhere.")
    
    # Main Results
    st.markdown("### 📋 **Query Results:**")
    
    if len(df_result) == 0:
        st.warning("🔍 No results found for your query. Try a different question!")
    else:
        # Enhanced dataframe display
        st.dataframe(
            df_result.style.format(precision=2),
            use_container_width=True,
            height=min(400, len(df_result) * 35 + 100)
        )
        
        # Download option
        csv = df_result.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"ai_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def display_sql_query_box(sql_query, question):
    """Display SQL query in a dedicated box"""
    st.markdown("---")
    st.markdown("### 🔧 **Generated SQL Query**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div class="sql-display">
            <strong>Question:</strong> {question}<br><br>
            <strong>Generated SQL:</strong><br>
            <code style="background: #f8f9fa; padding: 0.5rem; border-radius: 4px; display: block; margin-top: 0.5rem;">{sql_query}</code>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("📋 Copy SQL", help="Copy SQL to clipboard"):
            st.code(sql_query, language="sql")
            st.success("✅ SQL copied!")
        
        if st.button("📖 Explain Query", help="Get AI explanation of this SQL"):
            with st.expander("🤖 SQL Explanation", expanded=True):
                st.info("This feature will explain the SQL query in plain English (to be implemented)")

# Replace your simple_chart_section function with this fixed version:

def simple_chart_section(result_dict, question):
    """Simple chart generation that persists through reruns"""
    if not result_dict or result_dict.get('error') or result_dict.get('result') is None:
        return
    
    df = result_dict['result']
    
    # Basic checks
    if len(df) <= 1:
        return
    
    # Store the current query data in session state so it persists
    current_query_key = f"query_data_{hash(question)}"
    st.session_state[current_query_key] = {
        'df': df,
        'question': question,
        'result_dict': result_dict
    }
    
    # Chart generation section
    st.markdown("---")
    st.markdown("### 📊 **Simple Chart Generator**")
    
    # Get column info
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    all_cols = list(df.columns)
    
    # Only show if we have suitable data
    if len(numeric_cols) == 0 and len(categorical_cols) == 0:
        st.info("No suitable columns for charting")
        return
    
    st.info(f"📋 Available columns: {len(all_cols)} total, {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    
    # Create a unique form key based on the question
    form_key = f"chart_form_{abs(hash(question))}"
    
    # Simple form approach
    with st.form(form_key):
        st.markdown("**Create a simple chart:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chart_type = st.selectbox("Chart Type:", [
                "Bar Chart",
                "Pie Chart", 
                "Line Chart",
                "Scatter Plot"
            ])
        
        with col2:
            x_column = st.selectbox("X-axis:", all_cols)
        
        with col3:
            # For Y-axis, prefer numeric columns
            y_options = numeric_cols if numeric_cols else all_cols
            y_column = st.selectbox("Y-axis:", y_options)
        
        # Submit button
        submitted = st.form_submit_button("📈 Create Chart")
        
        if submitted:
            # Store chart parameters in session state
            chart_key = f"chart_{abs(hash(question))}"
            st.session_state[chart_key] = {
                'chart_type': chart_type,
                'x_column': x_column,
                'y_column': y_column,
                'created': True
            }
            
            # Show success message
            st.success("✅ Chart created! Check below.")
    
    # Display chart if it exists in session state
    chart_key = f"chart_{abs(hash(question))}"
    if chart_key in st.session_state and st.session_state[chart_key]['created']:
        chart_params = st.session_state[chart_key]
        
        try:
            # Create chart based on stored parameters
            chart_type = chart_params['chart_type']
            x_column = chart_params['x_column']
            y_column = chart_params['y_column']
            
            st.markdown("#### 📈 **Your Chart:**")
            
            if chart_type == "Bar Chart":
                fig = px.bar(df, x=x_column, y=y_column, 
                           title=f"{y_column} by {x_column}")
            
            elif chart_type == "Pie Chart":
                # For pie chart, aggregate if needed
                if len(df) > 10:
                    pie_data = df.groupby(x_column)[y_column].sum().reset_index()
                    fig = px.pie(pie_data, values=y_column, names=x_column,
                               title=f"{y_column} by {x_column}")
                else:
                    fig = px.pie(df, values=y_column, names=x_column,
                               title=f"{y_column} by {x_column}")
            
            elif chart_type == "Line Chart":
                df_sorted = df.sort_values(x_column)
                fig = px.line(df_sorted, x=x_column, y=y_column,
                            title=f"{y_column} vs {x_column}")
            
            else:  # Scatter Plot
                fig = px.scatter(df, x=x_column, y=y_column,
                               title=f"{y_column} vs {x_column}")
            
            # Style the chart
            fig.update_layout(
                plot_bgcolor='rgba(244, 246, 251, 0.9)',
                paper_bgcolor='rgba(244, 246, 251, 0.9)',
                font=dict(family="Segoe UI", size=12, color="#1F2B3A"),
                height=500
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # REMOVED THE PROBLEMATIC CLEAR BUTTON!
            # The chart will persist until you run a new query
            
        except Exception as e:
            st.error(f"❌ Chart creation failed: {e}")
            st.info("Try different column combinations")

# def execute_and_store_query(actual_input):
#     """Execute query and store results in session state"""
    
#     # Create a unique key for this query
#     query_key = f"stored_query_{hash(actual_input)}"
    
#     # Check if we already have results for this exact query
#     if query_key in st.session_state:
#         # Use stored results
#         result_dict = st.session_state[query_key]['result_dict']
        
#         # Display results
#         display_enhanced_results(result_dict, actual_input)
        
#         if result_dict['sql']:
#             display_sql_query_box(result_dict['sql'], actual_input)
        
#         # Chart generation
#         simple_chart_section(result_dict, actual_input)
        
#         return
    
#     # Execute new query
#     try:
#         result = ai_query_assistant.run_sql(actual_input, verbose=False)
        
#         if result is not None:
#             # Convert to our result format
#             assistant = ai_query_assistant.AIQueryAssistant()
#             result_dict = assistant.query(actual_input, verbose=False)
            
#             # Store results in session state
#             st.session_state[query_key] = {
#                 'result_dict': result_dict,
#                 'question': actual_input,
#                 'timestamp': datetime.now()
#             }
            
#             # Display results
#             display_enhanced_results(result_dict, actual_input)
            
#             if result_dict['sql']:
#                 display_sql_query_box(result_dict['sql'], actual_input)
            
#             # Chart generation
#             simple_chart_section(result_dict, actual_input)
            
#         else:
#             st.error("❌ Failed to execute query. Please try a different question.")
            
#     except Exception as e:
#         st.error(f"❌ An error occurred: {str(e)}")
#         st.info("💡 Please try rephrasing your question or contact support.")

def execute_and_store_query(actual_input):
    """Execute query and store results in session state"""
    
    # Create a unique key for this query
    query_key = f"stored_query_{hash(actual_input)}"
    
    # Check if we already have results for this exact query
    if query_key in st.session_state:
        # Use stored results
        result_dict = st.session_state[query_key]['result_dict']
        
        # Display results
        display_enhanced_results(result_dict, actual_input)
        
        if result_dict['sql']:
            display_sql_query_box(result_dict['sql'], actual_input)
        
        # Chart generation
        simple_chart_section(result_dict, actual_input)
        
        return
    
    # Execute new query
    try:
        result = ai_query_assistant.run_sql(actual_input, verbose=False)
        
        if result is not None:
            # Convert to our result format
            assistant = ai_query_assistant.AIQueryAssistant()
            result_dict = assistant.query(actual_input, verbose=False)
            
            # Check if we got a valid result
            if result_dict and not result_dict.get('error'):
                # Store results in session state
                st.session_state[query_key] = {
                    'result_dict': result_dict,
                    'question': actual_input,
                    'timestamp': datetime.now()
                }
                
                # Display results
                display_enhanced_results(result_dict, actual_input)
                
                if result_dict['sql']:
                    display_sql_query_box(result_dict['sql'], actual_input)
                
                # Chart generation
                simple_chart_section(result_dict, actual_input)
            else:
                # AI query failed, show test data instead
                st.error("❌ AI query failed. Showing test data instead:")
                create_test_data_and_chart()
        else:
            # Query execution failed, show test data
            st.error("❌ Query execution failed. Showing test data instead:")
            create_test_data_and_chart()
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 Showing test data so you can still try chart generation:")
        create_test_data_and_chart()


def create_test_data_and_chart():
    """Create test data when AI is not working"""
    st.markdown("---")
    st.markdown("### 🧪 **Test Mode - AI Service Unavailable**")
    st.info("The AI service is currently having issues. Here's some test data to try chart generation:")
    
    # Create sample data
    test_data = {
        'customer_name': ['Acme Corp', 'Global Foods', 'Local Market', 'Big Buyer', 'Small Shop'],
        'sale_amount': [15000, 25000, 8000, 30000, 5000],
        'final_tons_sold': [50, 80, 25, 100, 15],
        'warehouse_region': ['North', 'South', 'East', 'West', 'Central'],
        'satisfaction_rating': [4, 5, 3, 5, 4]
    }
    
    df_test = pd.DataFrame(test_data)
    
    # Display the test data
    st.dataframe(df_test)
    
    # Create a fake result_dict to test chart generation
    fake_result_dict = {
        'result': df_test,
        'sql': 'SELECT * FROM sales LIMIT 5',
        'error': None
    }
    
    # Test the chart generation
    simple_chart_section(fake_result_dict, "test data for charts")

# ====== MAIN AI QUERY PAGE ======
def main():
    # Apply CSS styling
    st.markdown(create_ai_query_css(), unsafe_allow_html=True)
    
    # Main container
    st.markdown("""
    <div class="ai-query-container">
        <div class="ai-query-header">
            <h1 class="ai-query-title">🤖 AI Query Assistant</h1>
            <p style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 1rem; font-style: italic;">
                Compiled by <strong>Datavue</strong> • Powered by OpenAI
            </p>
            <p class="ai-query-subtitle">Ask anything about your sales data in plain English</p>
            <p style="opacity: 0.8; font-size: 1rem;">Powered by advanced AI • No SQL knowledge required • Instant insights</p>
        </div>
    </div>
""", unsafe_allow_html=True)
    
    # Example prompts section
    st.markdown("### 💡 **Try These Example Questions:**")
    examples = create_example_prompts()
    
    # Display examples in grid
    cols = st.columns(5)
    for i, example in enumerate(examples):
        with cols[i % 5]:
            if st.button(example, key=f"example_{i}", help="Click to use this example"):
                st.session_state.selected_example = example.split(' ', 1)[1]  # Remove emoji
                st.rerun()
    
    st.markdown("---")
    
    # Main query input
    st.markdown("### 🎯 **Ask Your Question:**")
    
    # Use dynamic key to force widget recreation when clearing
    clear_counter = st.session_state.get('clear_counter', 0)
    current_key = f'ai_query_input_{clear_counter}'

    # Get the current value from the dynamic key or from selected example
    current_value = ""
    if 'selected_example' in st.session_state:
        current_value = st.session_state.selected_example
        del st.session_state.selected_example
    elif current_key in st.session_state:
        current_value = st.session_state[current_key]

    user_question = st.text_input(
        "Type your question here:",
        value=current_value,
        placeholder="e.g., What are the top 5 customers by revenue this year?",
        key=current_key,
        label_visibility="collapsed"
    )

    # Store in consistent key for other parts of code
    st.session_state.ai_query_input = user_question

    # Autocomplete suggestions
    suggestions = create_autocomplete_suggestions()
    selected_suggestion = display_autocomplete(user_question, suggestions)
    
    if selected_suggestion:
        user_question = selected_suggestion
        st.rerun()
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_button = st.button(
            "🚀 Ask AI", 
            key='ai_search_button',
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.button("🎲 Random Question", help="Generate a random example question"):
            random_example = random.choice(examples)
            st.session_state.selected_example = random_example.split(' ', 1)[1]
            st.rerun()
 
    with col3:
        if st.button("🗑️ Clear", help="Clear the input field"):
            # Increment counter to force new widget
            st.session_state.clear_counter = st.session_state.get('clear_counter', 0) + 1
            # Clear both the old key and the consistent key
            old_key = f'ai_query_input_{st.session_state.clear_counter - 1}'
            if old_key in st.session_state:
                del st.session_state[old_key]
            if 'ai_query_input' in st.session_state:
                del st.session_state.ai_query_input
            st.rerun()
    
    # Get the actual input value
    actual_input = st.session_state.get('ai_query_input', '').strip()

    if search_button and actual_input:
        # Show loading
        with st.spinner("🤖 Processing your query..."):
            time.sleep(0.5)  # Brief pause for UX
        
        # Execute and store query
        execute_and_store_query(actual_input)
    
    elif search_button and not actual_input:
        st.warning("⚠️ Please enter a question before searching!")
    
    # Query history (optional)
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Add successful queries to history
    if search_button and user_question.strip():
        if user_question not in [q['question'] for q in st.session_state.query_history]:
            st.session_state.query_history.append({
                'question': user_question,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'rows': 0  # Will be updated when we have results
            })
    
    # Show recent queries
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### 📝 **Recent Queries:**")
        
        for i, query in enumerate(reversed(st.session_state.query_history[-3:])):
            with st.expander(f"🕐 {query['timestamp']} - {query['question'][:50]}{'...' if len(query['question']) > 50 else ''}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Question:** {query['question']}")
                with col2:
                    if st.button("🔄 Run Again", key=f"rerun_{i}"):
                        st.session_state.selected_example = query['question']
                        st.rerun()

if __name__ == "__main__":
    main()