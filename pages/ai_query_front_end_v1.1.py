import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
import time
# import css_renderer
import ai_query_assistant

# Search feature
st.markdown("---")
st.subheader("🔍 Search Sales Data using AI Query?")

search_value = st.text_input(f"Search Sales Data using AI Query?", key='search_input')

search_button = st.button("Search", key='search_button')

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

def update_progress_bar(my_bar, time_delay=0.01):
    for percent_complete in range(100):
        time.sleep(time_delay)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()


if search_button:
    print(search_value)
    if search_value.strip() == "":
        update_progress_bar(my_bar, time_delay=0.001)
        st.warning("Please enter a customer name to search.")
    else:
        update_progress_bar(my_bar, time_delay=0.01)
        result_df = ai_query_assistant.run_sql(search_value)
        st.dataframe(result_df, use_container_width=True)