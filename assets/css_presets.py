import streamlit as st

def html_sidebar():
    st.html("""
        <style>
            
            section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]
            {
            color: #ffffff;}


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

            section[data-testid="stSidebar"] h1
            {
            color: #ffffff;
            font-size: 1.7rem;
            }

            section[data-testid="stSidebar"] p
            {
            color: #ffffff;
            # font-size: 1.7rem;
            }

        </style> """)
    
            
def html_header():
    st.html(""" <style>
                    header[data-testid="stHeader"] 
                    {
                    background: transparent;
                    visibility: hidden;
                    height: 0px;
                    color: white !important;
                    }
                </style>

    """)

def html_sidebar_clear_filters_btn():

    # Custom CSS for the Clear All Filters button in the sidebar
    st.sidebar.markdown("""                       


        <style>
           
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] button[data-testid="stBaseButton-secondary"] {
            background-color: white !important;
            border: 1px solid #ccc !important;
            transition: all 0.9s ease !important;
        }

        /* Basic button text state - black font */
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] button[data-testid="stBaseButton-secondary"] p {
            color: black !important;
        }

        /* Hover state - smooth transition with subtle changes */
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] button[data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(135deg, #66e6ff 0%, #418FDE 100%) !important;
        border-color: #adb5bd !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] button[data-testid="stBaseButton-secondary"]:hover p {
            color: black !important;
        }
                        
 
        </style>    
    """, unsafe_allow_html=True)    

def html_sidebar_nav_link():
    st.sidebar.markdown("""
        <style>
                        
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"]::before 
        {
                content: "Pages";
                display: block;
                color: white !important;
                font-weight: bold !important;
                font-size: 25px !important;
                padding: 16px 16px 8px 16px !important;
                margin-bottom: 8px !important;
        }
                        
        /* Navigation container */
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
            padding: 10px 0 !important;
        }

        /* Navigation links - normal state */
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLink"] {
            color: #e1e1e1 !important;
            text-decoration: none !important;
            padding: 8px 16px !important;
            margin: 2px 8px !important;
            border-radius: 6px !important;
            transition: all 0.3s ease !important;
            display: block !important;
            border-left: 3px solid transparent !important;
        }

        /* Navigation link text */
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLink"] span {
            color: #e1e1e1 !important;
            font-weight: 500 !important;
            text-transform: capitalize !important;
        }

        /* Navigation links - hover state */
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLink"]:hover {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-left-color: #66e6ff !important;
            transform: translateX(4px) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLink"]:hover span {
            color: #66e6ff !important;
        }

        /* Active/current page styling - you might need to add a class for this */
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLink"].active,
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLink"][aria-current="page"] {
            background: linear-gradient(135deg, #66e6ff20 0%, #418FDE20 100%) !important;
            border-left-color: #418FDE !important;
            border-left-width: 4px !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLink"].active span,
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLink"][aria-current="page"] span {
            color: #418FDE !important;
            font-weight: 600 !important;
        }

        # /* Navigation separator line */
        # section[data-testid="stSidebar"] div[data-testid="stSidebarNavSeparator"] {
        #     background-color: rgba(255, 255, 255, 0.2) !important;
        #     height: 1px !important;
        #     margin: 16px 12px !important;
        # }
                        
        /* Style the gray separator to match sidebar background */
        section[data-testid="stSidebar"] div[data-testid="stSidebarNavSeparator"] {
            background-color: #3f3c4d !important;
            height: 1px !important;
            margin: 8px 12px !important;
            border: none !important;
        }
                        
        /* Reduce spacing between navigation and user content */
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
                margin-bottom: -20px !important;
                padding-bottom: 10px !important;
        }
                        
        /* Reduce margin of Filters heading */
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] h1 {
            margin-top: 0px !important;
            margin-bottom: 16px !important;
        }
        </style>
    """, unsafe_allow_html=True)

def html_main_page():
    st.html("""
        <style>
            
           .stApp {
        background: linear-gradient(135deg, #66e6ff 0%, #418FDE 100%);
        }        
        
        div[data-testid="stMetric"] 
        {
        /* Your existing styles */
        background: linear-gradient(135deg, #66e6ff 0%, #418FDE 100%); ### 66e6ff 418FDE  ### #667eea 0%, #764ba2 
        # background: linear-gradient(35deg, #66e6ff, #418FDE)
        border: 2px solid #5a67d8;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);        
        /* Layout */
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        min-height: 150px;        

        /* Smooth transition for the expansion */          
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);        
        /* Prevent layout shift by reserving space for delta */
        overflow: hidden;           }

        /* Hide delta by default - make it invisible but keep space */
        div[data-testid="stMetricDelta"] {
            opacity: 0;
            transform: translateY(10px) scale(0.8);
            transition: all 0.3s ease;
            height: 0;
            margin: 0;
        }

        /* On HOVER: expand card and reveal delta */
        div[data-testid="stMetric"]:hover {
            transform: scale(1.02); /* 5% bigger */
            min-height: 180px; /* Slightly taller */
            padding: 2.5rem 1.5rem; /* More padding */
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        }

        div[data-testid="stMetric"]:hover div[data-testid="stMetricDelta"] {
            opacity: 1;
            transform: translateY(0) scale(1);
            height: auto;
            margin: 0.25rem 0;
        }


        div.stHorizontalBlock:nth-child(1) div[data-testid="stColumn"] p {
            color: #dce5ee;
            font-size: 1.1rem;
            font-style: italic;
            font-align: center;
            font-weight: 1;
            margin-bottom: 1.2rem;
            text-align: center;
        }
        div.stHorizontalBlock:nth-child(2) div[data-testid="stMetric"] * {
            color: white !important;
        }
        div.stHorizontalBlock:nth-child(2) div[data-testid="stColumn"]:nth-child(1) div[data-testid="stMetricDelta"] div {
        color: red !important; /* This changes both the arrow and text */
        }
        div.stHorizontalBlock:nth-child(2) div[data-testid="stColumn"]:nth-child(1) div[data-testid="stMetricDelta"] path {
        color: blue !important; /* This changes both the arrow and text */
        }
        
        /* Target Plotly chart containers for rounded corners */
        div[data-testid="stPlotlyChart"] > div {
            border-radius: 16px !important;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.20);
            border: 2px solid rgba(65, 143, 222, 0.2);
        }

        /* Ensure the SVG inside also respects the border radius */
        div[data-testid="stPlotlyChart"] svg {
            border-radius: 14px;
        }

        </style> """)