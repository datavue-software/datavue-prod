import pandas as pd
import numpy as np
from openai import OpenAI
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union



class AIInsightsAnalyzer:
    """
    AI-powered insights generator for data analysis
    Supports different models for different types of analysis
    """
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize the AI Insights Analyzer
        
        Args:
            api_key (str): OpenRouter API key
            base_url (str): API base URL (default: OpenRouter)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Different models for different purposes
        self.models = {
            'insights': 'anthropic/claude-3-haiku',  # Good for analytical insights
            'sql': 'mistralai/devstral-small:free',  # Good for code/SQL
            'creative': 'meta-llama/llama-3.1-8b-instruct:free',  # Good for creative content
            'fast': 'mistralai/devstral-small:free',  # Fast responses
            'detailed': 'anthropic/claude-3-sonnet',  # Detailed analysis (paid)
            'free': 'mistralai/devstral-small:free'  # Free tier model
        }
    
    def analyze_data_summary(self, df: pd.DataFrame, model_type: str = 'insights') -> str:
        """
        Generate high-level insights about the dataset
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            model_type (str): Type of model to use
            
        Returns:
            str: AI-generated insights
        """
        # Calculate basic statistics
        stats = {
            'total_records': len(df),
            'date_range': {
                'start': df['sale_date'].min().strftime('%Y-%m-%d'),
                'end': df['sale_date'].max().strftime('%Y-%m-%d')
            },
            'total_revenue': df['sale_amount'].sum(),
            'avg_order_value': df['sale_amount'].mean(),
            'total_customers': df['customer_name'].nunique(),
            'total_volume': df['final_tons_sold'].sum(),
            'avg_satisfaction': df['satisfaction_rating'].mean(),
            'top_region': df.groupby('warehouse_region')['sale_amount'].sum().idxmax(),
            'top_customer_category': df.groupby('customer_category')['sale_amount'].sum().idxmax()
        }
        
        prompt = f"""
You are a senior business analyst reviewing sales data. Provide 3-4 key insights in a conversational, professional tone.

**Dataset Overview:**
- Time Period: {stats['date_range']['start']} to {stats['date_range']['end']}
- Total Records: {stats['total_records']:,}
- Total Revenue: ${stats['total_revenue']:,.0f}
- Average Order Value: ${stats['avg_order_value']:,.0f}
- Unique Customers: {stats['total_customers']:,}
- Total Volume Sold: {stats['total_volume']:,.1f} tons
- Average Satisfaction: {stats['avg_satisfaction']:.1f}/5
- Top Performing Region: {stats['top_region']}
- Leading Customer Category: {stats['top_customer_category']}

Generate 3-4 bullet points highlighting the most important business insights. Focus on:
- Overall business performance
- Customer behavior patterns
- Operational strengths/concerns
- Actionable recommendations

Format as bullet points starting with relevant emojis.
"""
        
        return self._call_model(prompt, model_type)
    
    def analyze_chart_data(self, chart_data: Dict, chart_type: str, model_type: str = 'insights') -> str:
        """
        Generate insights about specific chart data
        
        Args:
            chart_data (Dict): Chart data to analyze
            chart_type (str): Type of chart (pie, line, bar, etc.)
            model_type (str): Model to use for analysis
            
        Returns:
            str: AI-generated insights about the chart
        """
        
        if chart_type == 'revenue_trend':
            return self._analyze_revenue_trend(chart_data, model_type)
        elif chart_type == 'customer_category_pie':
            return self._analyze_category_distribution(chart_data, model_type)
        elif chart_type == 'regional_performance':
            return self._analyze_regional_data(chart_data, model_type)
        elif chart_type == 'product_mix':
            return self._analyze_product_mix(chart_data, model_type)
        else:
            return self._analyze_generic_chart(chart_data, chart_type, model_type)
    
    def _analyze_revenue_trend(self, data: Dict, model_type: str) -> str:
        """Analyze revenue trend data"""
        
        # Assuming data has 'months' and 'revenue' lists
        months = data.get('months', [])
        revenues = data.get('revenues', [])
        
        if not months or not revenues:
            return "📊 Revenue data unavailable for analysis."
        
        # Calculate trend metrics
        revenue_change = ((revenues[-1] - revenues[0]) / revenues[0] * 100) if revenues[0] != 0 else 0
        avg_revenue = sum(revenues) / len(revenues)
        peak_month = months[revenues.index(max(revenues))]
        low_month = months[revenues.index(min(revenues))]
        
        prompt = f"""
Analyze this revenue trend data:

**Revenue Trend Analysis:**
- Time Period: {months[0]} to {months[-1]}
- Revenue Change: {revenue_change:.1f}% from start to end
- Average Monthly Revenue: ${avg_revenue:,.0f}
- Peak Performance: {peak_month} (${max(revenues):,.0f})
- Lowest Performance: {low_month} (${min(revenues):,.0f})
- Monthly Data: {dict(zip(months, [f'${r:,.0f}' for r in revenues]))}

Provide 2-3 insights about:
- Revenue growth trends and patterns
- Seasonal or cyclical patterns
- Performance highlights and concerns
- Recommendations for revenue optimization

Keep it concise and business-focused. Use emojis for visual appeal.
"""
        
        return self._call_model(prompt, model_type)
    
    def _analyze_category_distribution(self, data: Dict, model_type: str) -> str:
        """Analyze customer category pie chart data"""
        
        categories = data.get('categories', [])
        values = data.get('values', [])
        
        if not categories or not values:
            return "📊 Category distribution data unavailable."
        
        total = sum(values)
        percentages = [(v/total)*100 for v in values]
        top_category = categories[values.index(max(values))]
        
        prompt = f"""
Analyze this customer category distribution:

**Customer Category Breakdown:**
- Total Revenue: ${total:,.0f}
- Top Category: {top_category} (${max(values):,.0f}, {max(percentages):.1f}%)
- Distribution: {dict(zip(categories, [f'${v:,.0f} ({p:.1f}%)' for v, p in zip(values, percentages)]))}

Provide 2-3 insights about:
- Customer portfolio balance
- Revenue concentration risks
- Market segment opportunities
- Strategic recommendations

Use emojis and keep it actionable.
"""
        
        return self._call_model(prompt, model_type)
    
    def _analyze_regional_data(self, data: Dict, model_type: str) -> str:
        """Analyze regional performance data"""
        
        regions = data.get('regions', [])
        values = data.get('values', [])
        
        if not regions or not values:
            return "🗺️ Regional data unavailable for analysis."
        
        total = sum(values)
        top_region = regions[values.index(max(values))]
        weakest_region = regions[values.index(min(values))]
        
        prompt = f"""
Analyze this regional performance data:

**Regional Performance:**
- Total Revenue: ${total:,.0f}
- Top Performing Region: {top_region} (${max(values):,.0f})
- Underperforming Region: {weakest_region} (${min(values):,.0f})
- Regional Breakdown: {dict(zip(regions, [f'${v:,.0f}' for v in values]))}

Provide 2-3 insights about:
- Geographic revenue distribution
- Regional growth opportunities
- Market penetration strategies
- Resource allocation recommendations

Keep it strategic and actionable with emojis.
"""
        
        return self._call_model(prompt, model_type)
    
    def _analyze_product_mix(self, data: Dict, model_type: str) -> str:
        """Analyze product mix data"""
        
        products = data.get('products', [])
        volumes = data.get('volumes', [])
        
        if not products or not volumes:
            return "📦 Product mix data unavailable."
        
        total_volume = sum(volumes)
        top_product = products[volumes.index(max(volumes))]
        
        prompt = f"""
Analyze this product mix data:

**Product Performance:**
- Total Volume: {total_volume:,.1f} tons
- Leading Product: {top_product} ({max(volumes):,.1f} tons)
- Product Distribution: {dict(zip(products, [f'{v:,.1f} tons' for v in volumes]))}

Provide 2-3 insights about:
- Product portfolio performance
- Volume concentration and risks
- Product development opportunities
- Inventory and production recommendations

Use emojis and focus on operational insights.
"""
        
        return self._call_model(prompt, model_type)
    
    def _analyze_generic_chart(self, data: Dict, chart_type: str, model_type: str) -> str:
        """Generic chart analysis for custom charts"""
        
        prompt = f"""
Analyze this {chart_type} chart data:

**Data:** {json.dumps(data, indent=2)}

Provide 2-3 key insights about the patterns, trends, and business implications shown in this data.
Focus on actionable business intelligence. Use emojis for visual appeal.
"""
        
        return self._call_model(prompt, model_type)
    
    def generate_comparative_insights(self, current_data: Dict, previous_data: Dict, 
                                    model_type: str = 'insights') -> str:
        """
        Generate insights comparing current vs previous period data
        
        Args:
            current_data (Dict): Current period data
            previous_data (Dict): Previous period data for comparison
            model_type (str): Model to use
            
        Returns:
            str: Comparative insights
        """
        
        prompt = f"""
Compare these two time periods and provide key insights:

**Current Period:**
{json.dumps(current_data, indent=2)}

**Previous Period:**
{json.dumps(previous_data, indent=2)}

Provide 3-4 insights about:
- Performance changes (growth/decline)
- Trend shifts and patterns
- Key drivers of change
- Strategic recommendations

Focus on business impact and actionable insights. Use emojis.
"""
        
        return self._call_model(prompt, model_type)
    
    def generate_predictive_insights(self, historical_data: pd.DataFrame, 
                                   model_type: str = 'insights') -> str:
        """
        Generate forward-looking insights based on historical trends
        
        Args:
            historical_data (pd.DataFrame): Historical sales data
            model_type (str): Model to use
            
        Returns:
            str: Predictive insights and recommendations
        """
        
        # Calculate trend metrics
        monthly_revenue = historical_data.groupby(
            historical_data['sale_date'].dt.strftime('%Y-%m')
        )['sale_amount'].sum().reset_index()
        
        recent_avg = monthly_revenue.tail(3)['sale_amount'].mean()
        older_avg = monthly_revenue.head(3)['sale_amount'].mean()
        growth_trend = ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0
        
        prompt = f"""
Based on historical sales data, provide forward-looking business insights:

**Trend Analysis:**
- Data Period: {len(monthly_revenue)} months
- Recent 3-Month Average: ${recent_avg:,.0f}
- Earlier 3-Month Average: ${older_avg:,.0f}
- Growth Trend: {growth_trend:.1f}%
- Total Revenue: ${historical_data['sale_amount'].sum():,.0f}

Provide 3-4 forward-looking insights about:
- Revenue trajectory and predictions
- Seasonal patterns to expect
- Potential risks and opportunities
- Strategic recommendations for growth

Focus on actionable business intelligence with emojis.
"""
        
        return self._call_model(prompt, model_type)
    
    def _call_model(self, prompt: str, model_type: str) -> str:
        """
        Call the specified model with the given prompt
        
        Args:
            prompt (str): The prompt to send
            model_type (str): Type of model to use
            
        Returns:
            str: Model response
        """
        try:
            model_name = self.models.get(model_type, self.models['free'])
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a senior business analyst providing concise, actionable insights. Use emojis appropriately and focus on business value."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"❌ Analysis unavailable: {str(e)}"
    
    def get_available_models(self) -> Dict[str, str]:
        """Return available models and their purposes"""
        return self.models.copy()

# Example usage functions for Streamlit integration
def create_insights_analyzer(api_key: str) -> AIInsightsAnalyzer:
    """Create and return an insights analyzer instance"""
    return AIInsightsAnalyzer(api_key)

def get_dashboard_insights(df: pd.DataFrame, analyzer: AIInsightsAnalyzer, 
                          model_type: str = 'insights') -> str:
    """Get overall dashboard insights"""
    return analyzer.analyze_data_summary(df, model_type)

def get_chart_insights(chart_data: Dict, chart_type: str, 
                      analyzer: AIInsightsAnalyzer, model_type: str = 'insights') -> str:
    """Get insights for specific chart data"""
    return analyzer.analyze_chart_data(chart_data, chart_type, model_type)

# Example usage
if __name__ == "__main__":
    # Example usage
    API_KEY=""
    # try:
    #     import key
    #     API_KEY=key.OR_key
    # except ImportError:
    #     print("Need to get key.py file with API keys.")
    analyzer = AIInsightsAnalyzer(API_KEY)


    try:
        import streamlit as st
        API_KEY = st.secrets["OR_KEY"]
        st.write("Loaded API key from Streamlit secrets")
        st.write(f"key is {API_KEY[:10]} till {API_KEY[-10:]}")
    except Exception:
        print("No Streamlit secrets found, trying local key.py...")
        try:
            import key
            API_KEY = key.OR_key
        except ImportError:
            print("No API key found. Set st.secrets['OR_KEY'] or create key.py")
    
    # Load sample data
    df = pd.read_csv("partial_csv.csv", parse_dates=["sale_date"])
    
    # Get overall insights
    print("📊 DASHBOARD INSIGHTS:")
    print(analyzer.analyze_data_summary(df, model_type='insights'))
    
    # Example chart data analysis
    revenue_data = {
        'months': ['2024-01', '2024-02', '2024-03'],
        'revenues': [150000, 180000, 200000]
    }
    
    print("\n📈 REVENUE TREND INSIGHTS:")
    print(analyzer.analyze_chart_data(revenue_data, 'revenue_trend', 'insights'))
    
    # Show available models
    print("\n🤖 AVAILABLE MODELS:")
    for purpose, model in analyzer.get_available_models().items():
        print(f"- {purpose}: {model}")