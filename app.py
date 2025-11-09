import streamlit as st
import pandas as pd
from io import BytesIO
from streamlit_agraph import agraph, Node, Edge, Config
import json
import uuid
from datetime import datetime
import zipfile
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import requests
import networkx as nx
from collections import defaultdict
from io import BytesIO
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import sklearn
from rapidfuzz import fuzz, process
import json
import re
from rapidfuzz import fuzz, process


# Page configuration
st.set_page_config(
    page_title="AI Automation Business Analysis - Trade Data Cleaner made by Karan Koch",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Aarti Industries branding
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', 'Roboto', sans-serif;
    }
    
    /* Main background - Aarti blue theme */
    .stApp {
        background: linear-gradient(135deg, #0a2463 0%, #1e3a8a 30%, #2563eb 70%, #3b82f6 100%);
        background-attachment: fixed;
    }
    
    /* Top header bar with logo */
    .aarti-header {
        background: linear-gradient(90deg, #fb923c 0%, #f97316 50%, #ea580c 100%);
        padding: 1.5rem 2rem;
        margin: -5rem -5rem 2rem -5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        border-bottom: 4px solid #ffffff;
    }
    
    .aarti-logo {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .aarti-logo-circle {
        width: 60px;
        height: 60px;
        background: #ffffff;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .aarti-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: 0.5px;
    }
    
    .aarti-subtitle {
        color: #fef3c7;
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: -5px;
    }
    
    /* Header styling */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.4);
        margin-bottom: 0.5rem !important;
        letter-spacing: 1px;
    }
    
    h2 {
        color: #fbbf24 !important;
        font-weight: 600 !important;
        font-size: 1.7rem !important;
        margin-top: 2rem !important;
        border-bottom: 3px solid #f97316;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    h3 {
        color: #fde68a !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    
    /* Content containers - Bold white text */
    .stMarkdown p, .stMarkdown li {
        color: #ffffff !important;
        font-size: 1.1rem;
        line-height: 1.7;
        font-weight: 600 !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    
    /* Strong text */
    strong {
        color: #fbbf24 !important;
        font-weight: 600;
    }
    
    /* Input fields */
    .stTextInput input, .stMultiSelect, .stSelectbox {
        background-color: rgba(255, 255, 255, 0.98) !important;
        color: #0a2463 !important;
        border: 2px solid #f97316 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        padding: 0.7rem !important;
    }
    
    .stTextInput input:focus, .stMultiSelect:focus-within, .stSelectbox:focus-within {
        border-color: #fb923c !important;
        box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.2) !important;
    }
    
    /* Labels - Enhanced visibility */
    .stTextInput label, .stMultiSelect label, .stSelectbox label, [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    
    /* Info text and help text */
    .stTextInput > div > div > div, 
    .stMultiSelect > div > div > div,
    .stSelectbox > div > div > div,
    [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 500 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.4);
    }
    
    /* Success/Info messages - High contrast */
    .element-container .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Make all status text visible */
    .stMarkdown, .stText {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Buttons - Aarti orange theme */
    .stButton button {
        background: linear-gradient(135deg, #f97316 0%, #fb923c 50%, #fdba74 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 6px 20px rgba(249, 115, 22, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 50%, #fb923c 100%) !important;
        box-shadow: 0 8px 25px rgba(249, 115, 22, 0.7);
        transform: translateY(-3px) scale(1.02);
    }
    
    .stButton button:disabled {
        background: linear-gradient(135deg, #64748b 0%, #94a3b8 100%) !important;
        opacity: 0.5;
        box-shadow: none;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
        padding: 0.7rem 2.5rem !important;
        font-size: 1.05rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%) !important;
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.7);
    }
    
    /* Metrics - Aarti orange accent */
    [data-testid="stMetricValue"] {
        color: #fb923c !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: #fef3c7 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Info/Success/Warning boxes - High contrast white text */
    .stAlert {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.3) 0%, rgba(251, 146, 60, 0.25) 100%) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
        border-left: 5px solid #f97316 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        font-size: 1.1rem !important;
    }
    
    .stAlert > div {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.98) !important;
        border-radius: 12px;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
        border: 2px solid #f97316;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(251, 146, 60, 0.1) 100%);
        padding: 15px;
        border-radius: 12px;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(5px);
        border-radius: 10px;
        color: #fef3c7;
        font-weight: 600;
        padding: 14px 28px;
        border: 2px solid transparent;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f97316 0%, #fb923c 100%) !important;
        border-color: #fdba74;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.5);
        transform: scale(1.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(249, 115, 22, 0.3);
        border-color: #fb923c;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.2) 0%, rgba(251, 146, 60, 0.15) 100%) !important;
        border-radius: 10px !important;
        color: #fef3c7 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(251, 146, 60, 0.05) 100%);
        border-radius: 12px;
        border: 3px dashed rgba(249, 115, 22, 0.6);
        padding: 25px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #fb923c;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(251, 146, 60, 0.1) 100%);
    }
    
    /* Columns */
    [data-testid="column"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(249, 115, 22, 0.05) 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(249, 115, 22, 0.2);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #f97316 !important;
    }
    
    /* Multiselect dropdown */
    [data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.98) !important;
        border-radius: 10px;
    }
    
    /* Success message */
    .success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(52, 211, 153, 0.15) 100%) !important;
        color: #ffffff !important;
        border-left-color: #10b981 !important;
    }
    
    /* Section dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #f97316 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #f97316 0%, #fb923c 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%);
    }
</style>
""", unsafe_allow_html=True)

# Aarti Industries Header with custom logo
st.markdown("""
<div class="aarti-header">
    <div class="aarti-logo">
        <svg width="60" height="60" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <!-- Orange circle background -->
            <circle cx="50" cy="50" r="48" fill="#FFFFFF"/>
            <!-- Globe icon in Aarti blue -->
            <g transform="translate(50, 50)">
                <!-- Outer circle -->
                <circle cx="0" cy="0" r="35" fill="none" stroke="#0a2463" stroke-width="4"/>
                <!-- Vertical lines -->
                <line x1="0" y1="-35" x2="0" y2="35" stroke="#0a2463" stroke-width="3"/>
                <ellipse cx="0" cy="0" rx="15" ry="35" fill="none" stroke="#0a2463" stroke-width="3"/>
                <!-- Horizontal lines -->
                <line x1="-35" y1="0" x2="35" y2="0" stroke="#0a2463" stroke-width="3"/>
                <ellipse cx="0" cy="0" rx="35" ry="15" fill="none" stroke="#0a2463" stroke-width="3"/>
            </g>
        </svg>
        <div>
            <div class="aarti-title">AI Automation Business Analysis</div>
            <div class="aarti-subtitle">Trade Data Standardization System made by Karan Koch</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'current_mappings' not in st.session_state:
    st.session_state.current_mappings = {}
if 'current_files_hash' not in st.session_state:
    st.session_state.current_files_hash = None
if 'selected_data_type' not in st.session_state:
    st.session_state.selected_data_type = None

# Initialize session state for value chain
if 'value_chain_nodes' not in st.session_state:
    st.session_state.value_chain_nodes = []
if 'value_chain_edges' not in st.session_state:
    st.session_state.value_chain_edges = []
if 'node_counter' not in st.session_state:
    st.session_state.node_counter = 0
if 'selected_node_id' not in st.session_state:
    st.session_state.selected_node_id = None

# Initialize session state for saved datasets
if 'saved_datasets' not in st.session_state:
    st.session_state.saved_datasets = {}
if 'temp_cleaned_data' not in st.session_state:
    st.session_state.temp_cleaned_data = None
if 'temp_data_type' not in st.session_state:
    st.session_state.temp_data_type = None

# Initialize relation mappings in session state
if 'product_relations' not in st.session_state:
    st.session_state.product_relations = {}

def load_and_merge_files(uploaded_files):
    """Load and merge multiple Excel files into a single DataFrame"""
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_excel(file, engine='openpyxl')
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
    
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        return merged
    return None

def extract_commercial_name_import_export(description):
    """Extract commercial name from Import/Export Commercial Description field"""
    if pd.isna(description):
        return "Unknown"
    return str(description).strip()

def extract_commercial_name_global(description):
    """Extract commercial name from Global Trade Product Description field"""
    if pd.isna(description):
        return "Unknown"
    return str(description).strip()

def apply_mappings_import_export(df, mappings):
    """Apply name mappings and filter the DataFrame for Import/Export data"""
    if 'Commercial Description' not in df.columns:
        st.error("'Commercial Description' column not found in the data")
        return df
    
    df['Commercial Name'] = df['Commercial Description'].apply(extract_commercial_name_import_export)
    df['Standardized Name'] = df['Commercial Name'].map(mappings)
    filtered_df = df[df['Standardized Name'].notna()].copy()
    
    return filtered_df

def apply_mappings_global(df, mappings):
    """Apply name mappings and filter the DataFrame for Global Trade data"""
    if 'Product Description' not in df.columns:
        st.error("'Product Description' column not found in the data")
        return df
    
    df['Commercial Name'] = df['Product Description'].apply(extract_commercial_name_global)
    df['Standardized Name'] = df['Commercial Name'].map(mappings)
    filtered_df = df[df['Standardized Name'].notna()].copy()
    
    return filtered_df

def get_financial_year_quarter(date_val):
    """Convert date to FY Quarter (April-March cycle)"""
    try:
        if pd.isna(date_val):
            return None
        
        if isinstance(date_val, str):
            date_obj = pd.to_datetime(date_val, dayfirst=True, errors='coerce')
        else:
            date_obj = pd.to_datetime(date_val)
        
        if pd.isna(date_obj):
            return None
        
        year = date_obj.year
        month = date_obj.month
        
        # Determine FY and Quarter
        if month >= 4:  # April onwards
            fy_start = year
            fy_end = year + 1
        else:  # Jan-March
            fy_start = year - 1
            fy_end = year
        
        # Determine quarter (Q1=Apr-Jun, Q2=Jul-Sep, Q3=Oct-Dec, Q4=Jan-Mar)
        if month in [4, 5, 6]:
            quarter = "Q1"
        elif month in [7, 8, 9]:
            quarter = "Q2"
        elif month in [10, 11, 12]:
            quarter = "Q3"
        else:  # [1, 2, 3]
            quarter = "Q4"
        
        return f"FY{fy_start}-{str(fy_end)[2:]} {quarter}"
    except:
        return None

def process_dataset_for_analytics(df):
    """Process dataset: filter units, convert to MT, compute quarters"""
    # Find required columns
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'period' in col.lower():
            date_col = col
            break
    
    qty_col = None
    for col in df.columns:
        if 'quantity' in col.lower() or 'qty' in col.lower() or 'weight' in col.lower():
            qty_col = col
            break
    
    unit_col = None
    for col in df.columns:
        if 'unit' in col.lower() or 'uqc' in col.lower():
            unit_col = col
            break
    
    value_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'unit value' in col_lower and 'usd' in col_lower:
            value_col = col
            break
        elif 'unit price' in col_lower or 'price' in col_lower:
            value_col = col
            break
    
    if not value_col:
        for col in df.columns:
            col_lower = col.lower()
            if 'value' in col_lower or 'amount' in col_lower or 'fob' in col_lower or 'cif' in col_lower:
                value_col = col
                break
    
    # Find country/origin columns
    country_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'country' in col_lower or 'origin' in col_lower or 'destination' in col_lower:
            country_col = col
            break
    
    # Find supplier/buyer columns
    supplier_col = None
    buyer_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'supplier' in col_lower or 'exporter' in col_lower:
            supplier_col = col
        if 'buyer' in col_lower or 'importer' in col_lower or 'consignee' in col_lower:
            buyer_col = col
    
    if not all([date_col, qty_col, unit_col, value_col]):
        return None, f"Missing required columns. Found: Date={date_col}, Qty={qty_col}, Unit={unit_col}, Value={value_col}"
    
    # Create working copy
    df_work = df.copy()
    
    # Filter valid units (MT or KG only)
    df_work['Unit_Upper'] = df_work[unit_col].astype(str).str.upper().str.strip()
    valid_mask = df_work['Unit_Upper'].str.contains('METRIC TON|KILOGRAMS?|^MTS?$|^KGS?$', regex=True, na=False)
    df_work = df_work[valid_mask].copy()
    
    if len(df_work) == 0:
        return None, "No valid MT or KG rows found after filtering"
    
    # Convert all to MT
    df_work['Quantity_MT'] = np.where(
        df_work['Unit_Upper'].str.contains('KILOGRAM|^KGS?$', regex=True),
        df_work[qty_col].astype(float) / 1000,
        df_work[qty_col].astype(float)
    )
    
    # Convert price to $/kg
    df_work['Price_USD_per_kg'] = np.where(
        df_work['Unit_Upper'].str.contains('KILOGRAM|^KGS?$', regex=True),
        df_work[value_col].astype(float),
        df_work[value_col].astype(float) / 1000
    )
    
    # Add quarter column
    df_work['Quarter'] = df_work[date_col].apply(get_financial_year_quarter)
    df_work = df_work[df_work['Quarter'].notna()].copy()
    
    # Store column names for later use
    df_work.attrs['country_col'] = country_col
    df_work.attrs['supplier_col'] = supplier_col
    df_work.attrs['buyer_col'] = buyer_col
    
    return df_work, None

# Create tabs - NOW WITH 4 TABS
tab1, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📊 Data Processing",  "🔬 Value Chain Builder", "📈 Analytics & Insights", "📊Market Estimation", "📦EC analysis", "📄Report Generation", "🤖 Advanced Analytics" ])

# ========================================
# AI CLEANING HELPER FUNCTIONS
# Add these BEFORE the tab definitions (after imports)
# Place after: from io import BytesIO
# Place after: import plotly.graph_objects as go
# Add: import re
# Add: from rapidfuzz import fuzz, process
# ========================================


def ai_clean_data(df, product_name, synonyms_text, cas_number, desc_column, api_key=None, batch_size=50):
    """
    AI-powered data cleaning that filters product descriptions with strict exclusion rules
    """
    
    # Parse synonyms
    synonyms_list = [s.strip() for s in synonyms_text.split(',') if s.strip()] if synonyms_text else []
    
    # Normalize product name and synonyms
    product_normalized = product_name.strip().upper()
    synonyms_normalized = [s.upper() for s in synonyms_list]
    all_terms = [product_normalized] + synonyms_normalized
    
    # CRITICAL: Exclusion keywords for derivatives, salts, and formulations
    EXCLUDE_KEYWORDS = [
        # Chemical modifications
        'BROMO', 'CHLORO', 'NITRO', 'METHYL', 'ETHYL', 'PROPYL', 'BUTYL',
        'HYDROXY', 'AMINO', 'SULFO', 'CARBOXY', 'FLUORO', 'IODO',
        'DI-', 'TRI-', 'TETRA-', 'PENTA-', 'HEXA-', 'POLY-',
        'ORTHO-', 'META-', 'PARA-', 'ISO-', 'NEO-', 'SEC-', 'TERT-',
        
        # Salts and derivatives
        'HYDROCHLORIDE', 'SULFATE', 'NITRATE', 'ACETATE', 'PHOSPHATE',
        'CARBONATE', 'OXALATE', 'CITRATE', 'TARTRATE', 'BENZOATE',
        'CHLORIDE', 'BROMIDE', 'IODIDE', 'OXIDE',
        
        # Formulations and mixtures
        'SOLUTION', 'MIXTURE', 'COMPOUND', 'BLEND', 'FORMULATION',
        'EMULSION', 'SUSPENSION', 'DISPERSION', 'PREPARATION',
        'COMPOSITION', 'COMPLEX', 'DERIVATIVE', 'MODIFIED',
        
        # Related but different molecules
        'TOLUIDINE', 'PHENYLENEDIAMINE', 'DIPHENYLAMINE', 'NAPHTHYLAMINE',
        'BENZIDINE', 'XYLIDINE', 'CUMIDINE', 'MESIDINE'
    ]
    
    # Initialize results
    results = []
    
    # Technical keywords that indicate valid product mentions
    tech_keywords = [
        'ISO TANK', 'DRUMS', 'BULK', 'CONTAINER', 'TANKER', 
        'IBC', 'FLEXI', 'BAGS', 'PACKING', 'PACKAGE', 'SHIPMENT'
    ]
    
    # Purity indicators (only for base product)
    purity_patterns = [r'\d+%', r'\d+\.\d+%', r'PURE', r'GRADE', r'TECHNICAL', r'INDUSTRIAL']
    
    # Process each row
    for idx, row in df.iterrows():
        description = str(row[desc_column]).strip().upper()
        
        # Skip empty descriptions
        if not description or description == 'NAN':
            results.append({
                'index': idx,
                'keep': False,
                'confidence': 0.0,
                'reason': 'empty_description',
                'matched_term': None
            })
            continue
        
        # STEP 1: Check for exclusion keywords FIRST (highest priority)
        excluded = False
        exclusion_reason = None
        
        for exclude_word in EXCLUDE_KEYWORDS:
            if exclude_word in description:
                # Special case: if the exclude word is part of the target name itself, allow it
                if exclude_word in product_normalized:
                    continue
                
                # Check if it's modifying our target product (e.g., "BROMOANILINE" contains "ANILINE")
                # We need to be sure it's not just coincidental inclusion
                for term in all_terms:
                    # If term appears but with modification prefix/suffix, exclude it
                    if term in description:
                        # Check if exclude word is adjacent to term (within 3 characters)
                        term_pos = description.find(term)
                        exclude_pos = description.find(exclude_word)
                        
                        if abs(term_pos - exclude_pos) <= 20:  # Within 20 characters means likely related
                            excluded = True
                            exclusion_reason = f'derivative_or_salt_{exclude_word.lower()}'
                            break
                
                if excluded:
                    break
        
        if excluded:
            results.append({
                'index': idx,
                'keep': False,
                'confidence': 0.0,
                'reason': exclusion_reason,
                'matched_term': None
            })
            continue
        
        # STEP 2: Now check for positive matches (only if not excluded)
        keep = False
        confidence = 0.0
        reason = 'no_match'
        matched_term = None
        
        # 1. CAS number match (highest priority for positive match)
        if cas_number and cas_number.strip():
            cas_normalized = cas_number.strip().replace('-', '').replace(' ', '')
            desc_normalized = description.replace('-', '').replace(' ', '')
            if cas_normalized in desc_normalized:
                # Double check no exclusion words nearby
                cas_pos = desc_normalized.find(cas_normalized)
                surrounding_text = desc_normalized[max(0, cas_pos-30):min(len(desc_normalized), cas_pos+30)]
                
                excluded_near_cas = any(excl in surrounding_text for excl in EXCLUDE_KEYWORDS)
                if not excluded_near_cas:
                    keep = True
                    confidence = 0.99
                    reason = 'cas_match'
                    matched_term = cas_number
        
        # 2. Exact match with base product name
        if not keep:
            for term in all_terms:
                # Look for exact word boundary match
                # Use regex to ensure it's not part of a longer word
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, description):
                    keep = True
                    confidence = 0.95
                    reason = 'exact_match'
                    matched_term = term
                    break
        
        # 3. Technical specification + product name
        if not keep:
            for term in all_terms:
                if term in description:
                    for keyword in tech_keywords:
                        if keyword in description:
                            keep = True
                            confidence = 0.92
                            reason = 'tech_spec'
                            matched_term = term
                            break
                if keep:
                    break
        
        # 4. Purity specification + product name
        if not keep:
            for term in all_terms:
                if term in description:
                    for pattern in purity_patterns:
                        if re.search(pattern, description):
                            keep = True
                            confidence = 0.90
                            reason = 'purity_spec'
                            matched_term = term
                            break
                if keep:
                    break
        
        # 5. Repeated product name (indicates genuine mention)
        if not keep:
            for term in all_terms:
                if description.count(term) >= 2:
                    keep = True
                    confidence = 0.88
                    reason = 'repeated_name'
                    matched_term = term
                    break
        
        # 6. Fuzzy matching for typos (but be cautious)
        if not keep:
            for term in all_terms:
                # Only for reasonably long terms (avoid false matches)
                if len(term) >= 6:
                    ratio = fuzz.ratio(term, description)  # Use full ratio, not partial
                    if ratio >= 90:  # Strict threshold
                        keep = True
                        confidence = ratio / 100.0
                        reason = 'fuzzy_match'
                        matched_term = term
                        break
        
        # STEP 3: Final validation - recheck for exclusions even if matched
        if keep:
            # Scan entire description again for any exclusion words
            # that might have been missed
            words_in_desc = description.split()
            for word in words_in_desc:
                for exclude_word in EXCLUDE_KEYWORDS:
                    if exclude_word in word and len(word) > len(exclude_word):
                        # This word contains an exclusion keyword and is longer
                        # Likely a derivative (e.g., "BROMOANILINE")
                        keep = False
                        reason = 'derivative_detected'
                        confidence = 0.0
                        break
                if not keep:
                    break
        
        results.append({
            'index': idx,
            'keep': keep,
            'confidence': confidence,
            'reason': reason,
            'matched_term': matched_term
        })
    
    # If API key provided, enhance with AI
    if api_key:
        results = enhance_with_ai(results, df, desc_column, product_name, synonyms_list, cas_number, api_key, batch_size)
    
    # Split into cleaned and rejected
    kept_indices = [r['index'] for r in results if r['keep']]
    rejected_indices = [r['index'] for r in results if not r['keep']]
    
    cleaned_df = df.loc[kept_indices].copy()
    rejected_df = df.loc[rejected_indices].copy()
    
    # Add confidence scores and metadata
    for r in results:
        if r['keep']:
            cleaned_df.loc[r['index'], '_match_confidence'] = r['confidence']
            cleaned_df.loc[r['index'], '_match_pattern'] = r['reason']
            cleaned_df.loc[r['index'], '_matched_term'] = str(r['matched_term']) if r['matched_term'] else ''
        else:
            rejected_df.loc[r['index'], '_match_confidence'] = r['confidence']
            rejected_df.loc[r['index'], '_match_pattern'] = r['reason']
    
    # Calculate statistics
    stats = {
        'original': len(df),
        'cleaned': len(cleaned_df),
        'removed': len(rejected_df),
        'kept_percentage': (len(cleaned_df) / len(df) * 100) if len(df) > 0 else 0,
        'pattern_distribution': {},
        'exclusion_stats': {}
    }
    
    # Pattern distribution for kept records
    for r in results:
        if r['keep']:
            stats['pattern_distribution'][r['reason']] = stats['pattern_distribution'].get(r['reason'], 0) + 1
    
    # Exclusion distribution
    for r in results:
        if not r['keep'] and r['reason'] != 'no_match' and r['reason'] != 'empty_description':
            stats['exclusion_stats'][r['reason']] = stats['exclusion_stats'].get(r['reason'], 0) + 1
    
    # Confidence scores
    kept_confidences = [r['confidence'] for r in results if r['keep']]
    confidence_scores = {
        'mean': sum(kept_confidences) / len(kept_confidences) if kept_confidences else 0,
        'median': sorted(kept_confidences)[len(kept_confidences)//2] if kept_confidences else 0,
        'min': min(kept_confidences) if kept_confidences else 0,
        'max': max(kept_confidences) if kept_confidences else 0
    }
    
    return cleaned_df, rejected_df, stats, confidence_scores





def enhance_with_ai(results, df, desc_column, product_name, synonyms, cas_number, api_key, batch_size=50):
    """
    Enhance filtering results using Perplexity AI for uncertain cases
    """
    import requests
    
    # Find uncertain cases (confidence < 0.7)
    uncertain_indices = [i for i, r in enumerate(results) if not r['keep'] or r['confidence'] < 0.7]
    
    if not uncertain_indices:
        return results
    
    # Process in batches
    for batch_start in range(0, len(uncertain_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(uncertain_indices))
        batch_indices = uncertain_indices[batch_start:batch_end]
        
        # Get descriptions for this batch
        batch_descriptions = [
            str(df.iloc[results[i]['index']][desc_column]).strip() 
            for i in batch_indices
        ]
        
        # Call AI
        try:
            ai_results = ai_filter_descriptions_batch(
                batch_descriptions, 
                product_name, 
                synonyms, 
                cas_number, 
                api_key
            )
            
            # Update results
            for i, desc in enumerate(batch_descriptions):
                result_idx = batch_indices[i]
                if desc in ai_results:
                    ai_result = ai_results[desc]
                    results[result_idx]['keep'] = ai_result['keep']
                    results[result_idx]['confidence'] = ai_result['confidence']
                    results[result_idx]['reason'] = ai_result['reason']
        
        except Exception as e:
            continue
    
    return results


def export_rejected_data(rejected_df, product_name, desc_column):
    """
    Export rejected data to Excel for manual review
    """
    from io import BytesIO
    import pandas as pd
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main rejected data
        rejected_df.to_excel(writer, sheet_name='Rejected Entries', index=False)
        
        # Summary sheet
        summary_data = {
            'Metric': ['Total Rejected', 'Product Searched', 'Date Generated'],
            'Value': [
                len(rejected_df),
                product_name,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Pattern distribution
        if '_match_pattern' in rejected_df.columns:
            pattern_counts = rejected_df['_match_pattern'].value_counts()
            pattern_df = pd.DataFrame({
                'Pattern': pattern_counts.index,
                'Count': pattern_counts.values
            })
            pattern_df.to_excel(writer, sheet_name='Rejection Patterns', index=False)
    
    output.seek(0)
    return output


# ========================================
# COMPLETE TAB 1 - DATA PROCESSING WITH AI CLEANING
# Replace your existing with tab1: section with this complete code
# ========================================

with tab1:
    st.header("📦 Trade Data Processing")
    
    # Data type selector
    data_type = st.selectbox(
        "Select Data Type",
        options=["Import", "Export", "Global"],
        help="Choose the type of trade data you want to process",
        key="data_type_selector"
    )
    
    # Get icon based on data type
    data_type_icon = {
        "Import": "📥",
        "Export": "📤",
        "Global": "🌍"
    }
    
    st.markdown(f"### {data_type_icon[data_type]} Processing {data_type} Trade Data")
    
    # Check if data type changed
    if st.session_state.selected_data_type != data_type:
        st.session_state.selected_data_type = data_type
        st.session_state.current_df = None
        st.session_state.current_mappings = {}
        st.session_state.current_files_hash = None
    
    # Determine the description column based on data type
    desc_column = 'Product Description' if data_type == 'Global' else 'Commercial Description'
    apply_func = apply_mappings_global if data_type == 'Global' else apply_mappings_import_export
    extract_func = extract_commercial_name_global if data_type == 'Global' else extract_commercial_name_import_export
    
    # ========================================
    # CLEANING MODE SELECTOR
    # ========================================
    st.markdown("---")
    st.markdown("### 🧹 Data Cleaning Mode")
    
    col_mode1, col_mode2 = st.columns([1, 2])
    
    with col_mode1:
        cleaning_mode = st.selectbox(
            "Select Cleaning Method",
            options=["Manual Cleaning", "AI-Powered Cleaning"],
            help="Choose between manual filtering or AI-powered automatic cleaning",
            key="cleaning_mode_selector"
        )
    
    with col_mode2:
        if cleaning_mode == "AI-Powered Cleaning":
            st.info("🤖 AI will automatically filter product descriptions based on your inputs")
        else:
            st.info("👤 You will manually map and filter commercial names")
    
    # ========================================
    # AI CLEANING CONFIGURATION
    # ========================================
    if cleaning_mode == "AI-Powered Cleaning":
        st.markdown("---")
        st.markdown("#### 🎯 AI Cleaning Configuration")
        
        col_ai1, col_ai2, col_ai3 = st.columns(3)
        
        with col_ai1:
            ai_product_name = st.text_input(
                "Product Name *",
                placeholder="e.g., Aniline",
                help="Enter the main product name to search for",
                key="ai_product_name"
            )
        
        with col_ai2:
            ai_synonyms = st.text_input(
                "Synonyms (comma-separated)",
                placeholder="e.g., Phenylamine, Aminobenzene",
                help="Enter alternative names for the product",
                key="ai_synonyms"
            )
        
        with col_ai3:
            ai_cas_number = st.text_input(
                "CAS Number",
                placeholder="e.g., 62-53-3",
                help="Enter the CAS registry number",
                key="ai_cas_number"
            )
        
        # Advanced options
        with st.expander("⚙️ Advanced AI Options"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                use_ai_api = st.checkbox(
                    "Use Perplexity AI Enhancement",
                    value=False,
                    help="Enable AI for uncertain cases (requires API key)"
                )
                
                if use_ai_api:
                    api_key_input = st.text_input(
                        "Perplexity API Key",
                        type="password",
                        value="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",
                        help="Enter your Perplexity API key"
                    )
            
            with col_adv2:
                batch_size = st.number_input(
                    "AI Batch Size",
                    min_value=10,
                    max_value=100,
                    value=50,
                    help="Number of descriptions to process per API call"
                )
                
                st.info("""
                **AI Cleaning Features:**
                - ✅ Exact name matching
                - ✅ CAS number detection
                - ✅ Fuzzy matching for typos
                - ✅ Technical spec recognition
                - ✅ Synonym matching
                - ✅ Derivative detection
                - 🤖 AI enhancement (optional)
                """)
    
    # ========================================
    # FILE UPLOADER
    # ========================================
    st.markdown("---")
    files = st.file_uploader(
        f"Select {data_type} Excel files (.xlsx)",
        type=['xlsx'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if files:
        # Create a hash of uploaded files to detect changes
        files_hash = hash(tuple(f.name for f in files))
        
        # Reset mappings if new files are uploaded
        if files_hash != st.session_state.current_files_hash:
            st.session_state.current_mappings = {}
            st.session_state.current_files_hash = files_hash
    
    # ========================================
    # LOAD AND MERGE FILES WITH AI CLEANING
    # ========================================
    st.markdown("---")
    if st.button(f"Load and Merge {data_type} Files", key="load_button") or st.session_state.current_df is not None:
        if st.session_state.current_df is None:
            with st.spinner(f"Loading and merging {data_type.lower()} files..."):
                st.session_state.current_df = load_and_merge_files(files)
        
        if st.session_state.current_df is not None:
            df = st.session_state.current_df
            
            # Check if required column exists
            if desc_column in df.columns:
                
                # ========================================
                # AI CLEANING EXECUTION
                # ========================================
                if cleaning_mode == "AI-Powered Cleaning":
                    if not ai_product_name:
                        st.error("⚠️ Please enter a Product Name for AI cleaning")
                    else:
                        st.markdown("---")
                        st.markdown("### 🤖 AI-Powered Cleaning")
                        
                        with st.spinner("🔄 Running AI-powered data cleaning..."):
                            # Run AI cleaning
                            cleaned_df, rejected_df, stats, confidence_scores = ai_clean_data(
                                df=df.copy(),
                                product_name=ai_product_name,
                                synonyms_text=ai_synonyms if ai_synonyms else "",
                                cas_number=ai_cas_number if ai_cas_number else "",
                                desc_column=desc_column,
                                api_key=api_key_input if use_ai_api else None,
                                batch_size=batch_size if use_ai_api else 50
                            )
                            
                            # Replace current dataframe with cleaned version
                            df = cleaned_df
                        
                        # Show cleaning results
                        st.success(f"✅ AI Cleaning Complete!")
                        
                        # Main statistics
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        with col_stat1:
                            st.metric("Original Rows", f"{stats['original']:,}")
                        with col_stat2:
                            st.metric("Kept Rows", f"{stats['cleaned']:,}")
                        with col_stat3:
                            st.metric("Removed Rows", f"{stats['removed']:,}")
                        with col_stat4:
                            st.metric("Success Rate", f"{stats['kept_percentage']:.1f}%")
                        
                        # Confidence scores visualization
                        st.markdown("---")
                        st.markdown("#### 📊 Confidence Score Analysis")
                        
                        col_conf1, col_conf2, col_conf3, col_conf4 = st.columns(4)
                        with col_conf1:
                            st.metric("Mean Confidence", f"{confidence_scores['mean']:.2%}")
                        with col_conf2:
                            st.metric("Median Confidence", f"{confidence_scores['median']:.2%}")
                        with col_conf3:
                            st.metric("Min Confidence", f"{confidence_scores['min']:.2%}")
                        with col_conf4:
                            st.metric("Max Confidence", f"{confidence_scores['max']:.2%}")
                        
                        # Pattern distribution
                        if 'pattern_distribution' in stats and stats['pattern_distribution']:
                            st.markdown("#### 🔍 Pattern Recognition Analysis")
                            
                            pattern_dist = stats['pattern_distribution']
                            
                            # Create pattern explanation mapping
                            pattern_explanations = {
                                'exact_match': '🎯 Exact product name match',
                                'cas_match': '🔬 CAS number detected',
                                'tech_spec': '📦 Technical specifications (ISO TANK, DRUMS, etc.)',
                                'repeated_name': '🔄 Repeated product name',
                                'purity_spec': '💎 Purity specification (99%, etc.)',
                                'formula_match': '⚗️ Chemical formula match',
                                'prefix_match': '📝 Prefix/suffix variation (PARA-, ORTHO-, etc.)',
                                'synonym_match': '📚 Synonym detected',
                                'typo_match': '✏️ Minor typo corrected',
                                'fuzzy_match': '🔎 Fuzzy string matching',
                                'ai_classified': '🤖 AI classification',
                                'derivative_uncertain': '⚠️ Possible derivative (low confidence)'
                            }
                            
                            # Create pattern chart
                            pattern_df = pd.DataFrame({
                                'Pattern': list(pattern_dist.keys()),
                                'Count': list(pattern_dist.values())
                            }).sort_values('Count', ascending=False)
                            
                            fig_pattern = go.Figure(data=[
                                go.Bar(
                                    x=pattern_df['Count'],
                                    y=pattern_df['Pattern'],
                                    orientation='h',
                                    marker_color='#3b82f6',
                                    text=pattern_df['Count'],
                                    textposition='auto',
                                )
                            ])
                            
                            fig_pattern.update_layout(
                                title="Pattern Type Distribution",
                                xaxis_title="Number of Matches",
                                yaxis_title="Pattern Type",
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_pattern, use_container_width=True)
                            
                            # Pattern explanations
                            with st.expander("📖 Pattern Type Explanations"):
                                for _, row in pattern_df.iterrows():
                                    pattern_name = row['Pattern']
                                    pattern_count = row['Count']
                                    explanation = pattern_explanations.get(pattern_name, pattern_name)
                                    st.write(f"**{explanation}**: {pattern_count} matches")
                        
                        # Rejected data export
                        st.markdown("---")
                        st.markdown("#### 📥 Export Rejected Entries")
                        
                        col_exp1, col_exp2 = st.columns([2, 1])
                        
                        with col_exp1:
                            st.info(f"💡 {len(rejected_df)} entries were filtered out. Download them for manual review.")
                        
                        with col_exp2:
                            if len(rejected_df) > 0:
                                rejected_excel = export_rejected_data(
                                    rejected_df, 
                                    ai_product_name, 
                                    desc_column
                                )
                                
                                if rejected_excel:
                                    st.download_button(
                                        label="📥 Download Rejected",
                                        data=rejected_excel,
                                        file_name=f"rejected_{ai_product_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                        
                        # Sample of rejected entries
                        if len(rejected_df) > 0:
                            with st.expander(f"👁️ Preview Rejected Entries (showing {min(20, len(rejected_df))} of {len(rejected_df)})"):
                                display_cols = [desc_column, '_match_confidence', '_match_pattern']
                                display_cols = [col for col in display_cols if col in rejected_df.columns]
                                
                                preview_df = rejected_df[display_cols].head(20).copy()
                                
                                if '_match_confidence' in preview_df.columns:
                                    preview_df['_match_confidence'] = preview_df['_match_confidence'].apply(
                                        lambda x: f"{x:.2%}" if isinstance(x, float) else x
                                    )
                                
                                preview_df = preview_df.rename(columns={
                                    desc_column: 'Description',
                                    '_match_confidence': 'Confidence',
                                    '_match_pattern': 'Pattern'
                                })
                                
                                st.dataframe(preview_df, use_container_width=True)
                
                # ========================================
                # MANUAL CLEANING MODE
                # ========================================
                else:
                    # Manual cleaning mode - existing logic
                    st.subheader("🔍 Pre-Filter Data (Optional)")
                    st.write("Filter out unwanted entries before extracting unique names")
                    
                    col_filter1, col_filter2 = st.columns(2)
                    with col_filter1:
                        exclude_keywords = st.text_input(
                            "🚫 Exclude entries containing (comma-separated)",
                            placeholder="e.g., TESTING, SAMPLE, TRIAL",
                            key="pre_exclude",
                            help="Remove rows where description contains any of these keywords"
                        )
                    with col_filter2:
                        include_keywords = st.text_input(
                            "✓ Include only entries containing (comma-separated)",
                            placeholder="e.g., ANILINE, CHEMICAL",
                            key="pre_include",
                            help="Keep only rows where description contains any of these keywords (leave empty to include all)"
                        )
                    
                    # Apply pre-filtering
                    filtered_df = df.copy()
                    original_count = len(filtered_df)
                    
                    # Exclude filter
                    if exclude_keywords:
                        exclude_list = [kw.strip().upper() for kw in exclude_keywords.split(',') if kw.strip()]
                        for keyword in exclude_list:
                            filtered_df = filtered_df[~filtered_df[desc_column].astype(str).str.upper().str.contains(keyword, na=False)]
                    
                    # Include filter
                    if include_keywords:
                        include_list = [kw.strip().upper() for kw in include_keywords.split(',') if kw.strip()]
                        mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
                        for keyword in include_list:
                            mask = mask | filtered_df[desc_column].astype(str).str.upper().str.contains(keyword, na=False)
                        filtered_df = filtered_df[mask]
                    
                    rows_removed = original_count - len(filtered_df)
                    
                    if exclude_keywords or include_keywords:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Rows", original_count)
                        with col2:
                            st.metric("Filtered Rows", len(filtered_df))
                        with col3:
                            st.metric("Removed", rows_removed, delta=f"-{rows_removed}")
                    
                    # Extract unique names from filtered data
                    filtered_df['Commercial Name'] = filtered_df[desc_column].apply(extract_func)
                    unique_names = sorted(filtered_df['Commercial Name'].unique())
                    
                    st.success(f"✅ Loaded {len(filtered_df)} rows from {len(files)} file(s)")
                    st.info(f"Found {len(unique_names)} unique commercial names")
                    
                    # Update the working dataframe to filtered version
                    df = filtered_df
                
                # ========================================
                # DISPLAY SAMPLE DATA (COMMON)
                # ========================================
                with st.expander(f"📊 Preview {data_type} Data"):
                    st.dataframe(df.head(20))
                
                # ========================================
                # COMMERCIAL NAME STANDARDIZATION (MANUAL MODE ONLY)
                # ========================================
                if cleaning_mode == "Manual Cleaning":
                    st.subheader(f"🔄 {data_type} Name Standardization")
                    
                    # Display current mappings
                    if st.session_state.current_mappings:
                        st.write("**Current Mappings:**")
                        mapping_df = pd.DataFrame([
                            {"Source Name": k, "→ Target Name": v}
                            for k, v in st.session_state.current_mappings.items()
                        ])
                        st.dataframe(mapping_df, use_container_width=True)
                        
                        if st.button(f"🗑️ Clear All {data_type} Mappings", key="clear_button"):
                            st.session_state.current_mappings = {}
                            st.rerun()
                    
                    # Get unmapped names
                    unmapped_names = [name for name in unique_names 
                                     if name not in st.session_state.current_mappings]
                    
                    if unmapped_names:
                        st.write("**Create New Mapping:**")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Add Select All checkbox
                            select_all = st.checkbox(
                                "Select All Unmapped Names",
                                key="select_all_checkbox",
                                help="Select all unmapped names at once"
                            )
                            
                            selected_names = st.multiselect(
                                "Select Source Names (original names to standardize)",
                                options=unmapped_names,
                                default=unmapped_names if select_all else [],
                                help="Select one or more names to map to a standardized name",
                                key="sources"
                            )
                        
                        with col2:
                            target_name = st.text_input(
                                "Target Name (standardized name)",
                                help="Enter the generalized/standardized name",
                                key="target"
                            )
                        
                        if st.button(f"➕ Add {data_type} Mapping", 
                                   disabled=not (selected_names and target_name),
                                   key="add_button"):
                            for name in selected_names:
                                st.session_state.current_mappings[name] = target_name
                            st.success(f"Added mapping: {len(selected_names)} name(s) → {target_name}")
                            st.rerun()
                    else:
                        st.success(f"✅ All unique {data_type.lower()} names have been mapped!")
                    
                    # Show mapping statistics
                    st.metric(f"Mapped {data_type} Names", f"{len(st.session_state.current_mappings)} / {len(unique_names)}")
                
                # ========================================
                # APPLY MAPPINGS AND GENERATE OUTPUT (MANUAL MODE ONLY)
                # ========================================
                if cleaning_mode == "Manual Cleaning" and st.session_state.current_mappings:
                    st.subheader(f"📥 Generate Cleaned {data_type} Data")
                    
                    if st.button(f"🔄 Apply {data_type} Mappings and Filter Data", key="apply_button"):
                        with st.spinner("Processing..."):
                            cleaned_df = apply_func(df.copy(), st.session_state.current_mappings)
                            
                            # Store cleaned data in session state temporarily
                            st.session_state.temp_cleaned_data = cleaned_df
                            st.session_state.temp_data_type = data_type
                            
                            st.success(f"✅ Processed! {len(cleaned_df)} rows retained (filtered from {len(df)} original rows)")
                            
                            # Display results
                            st.write(f"**Cleaned {data_type} Data Preview:**")
                            st.dataframe(cleaned_df.head(20))
                            
                            # Download button
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                cleaned_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
                            output.seek(0)
                            
                            st.download_button(
                                label=f"📥 Download Cleaned {data_type} Data",
                                data=output,
                                file_name=f"cleaned_{data_type.lower()}_trade_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_button"
                            )
                            
                            # Statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Original Rows", len(df))
                            with col2:
                                st.metric("Cleaned Rows", len(cleaned_df))
                            with col3:
                                st.metric("Removed Rows", len(df) - len(cleaned_df))
                
                # ========================================
                # FOR AI MODE, STORE CLEANED DATA AUTOMATICALLY
                # ========================================
                elif cleaning_mode == "AI-Powered Cleaning" and 'cleaned_df' in locals():
                    st.session_state.temp_cleaned_data = df
                    st.session_state.temp_data_type = data_type
                
                # ========================================
                # SAVE DATASET SECTION (COMMON FOR BOTH MODES)
                # ========================================
                if st.session_state.temp_cleaned_data is not None:
                    st.markdown("---")
                    st.subheader("💾 Save Cleaned Dataset")
                    st.write("Save this cleaned dataset for future analytics and processing")
                    
                    col_name, col_save = st.columns([3, 1])
                    
                    with col_name:
                        dataset_name = st.text_input(
                            "Enter Dataset Name",
                            placeholder=f"e.g., {data_type}_Q1_2024_Cleaned",
                            key="dataset_name_input",
                            help="Give this dataset a unique, descriptive name"
                        )
                    
                    with col_save:
                        st.write("")  # Spacing
                        st.write("")  # Spacing
                        if st.button("💾 Save Dataset", key="save_dataset_button", disabled=not dataset_name):
                            if dataset_name in st.session_state.saved_datasets:
                                st.warning(f"⚠️ Dataset '{dataset_name}' already exists! Choose a different name.")
                            else:
                                # Save dataset with metadata
                                st.session_state.saved_datasets[dataset_name] = {
                                    'data': st.session_state.temp_cleaned_data.copy(),
                                    'type': st.session_state.temp_data_type,
                                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'rows': len(st.session_state.temp_cleaned_data),
                                    'columns': list(st.session_state.temp_cleaned_data.columns)
                                }
                                
                                st.success(f"✅ Dataset '{dataset_name}' saved successfully!")
                                st.balloons()
                                
                                # Clear temp data
                                st.session_state.temp_cleaned_data = None
                                st.rerun()
            else:
                st.error(f"'{desc_column}' column not found in the uploaded files")
    else:
        st.info(f"👆 Upload {data_type} Excel files to begin")
    
    # ========================================
    # DISPLAY SAVED DATASETS
    # ========================================
    if st.session_state.saved_datasets:
        st.markdown("---")
        st.subheader("📚 Saved Datasets")
        st.write(f"Total saved datasets: **{len(st.session_state.saved_datasets)}**")
        
        # Create a summary table
        saved_summary = []
        for name, info in st.session_state.saved_datasets.items():
            saved_summary.append({
                "Dataset Name": name,
                "Type": info['type'],
                "Rows": info['rows'],
                "Saved On": info['date']
            })
        
        summary_df = pd.DataFrame(saved_summary)
        st.dataframe(summary_df, use_container_width=True)
        
        # Dataset management
        col_manage1, col_manage2 = st.columns(2)
        
        with col_manage1:
            selected_dataset = st.selectbox(
                "Select dataset to manage:",
                options=list(st.session_state.saved_datasets.keys()),
                key="manage_dataset_select"
            )
        
        with col_manage2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            col_view, col_delete = st.columns(2)
            
            with col_view:
                if st.button("👁️ View", key="view_dataset_button"):
                    if selected_dataset:
                        dataset_info = st.session_state.saved_datasets[selected_dataset]
                        st.write(f"**Dataset: {selected_dataset}**")
                        st.write(f"Type: {dataset_info['type']} | Rows: {dataset_info['rows']} | Date: {dataset_info['date']}")
                        st.dataframe(dataset_info['data'].head(50))
            
            with col_delete:
                if st.button("🗑️ Delete", key="delete_dataset_button"):
                    if selected_dataset:
                        del st.session_state.saved_datasets[selected_dataset]
                        st.success(f"Deleted dataset: {selected_dataset}")
                        st.rerun()





with tab3:
    st.markdown("""
    ### 🔬 Chemical Value Chain Builder
    Build interactive value chains for chemical molecules and their relationships.
    """)
    
    # Create two columns for controls and visualization
    col_controls, col_viz = st.columns([1, 2])
    
    with col_controls:
        st.markdown("#### 🎛️ Controls")
        
        # Add new node section
        with st.expander("➕ Add New Molecule", expanded=True):
            new_molecule_name = st.text_input(
                "Molecule Name",
                placeholder="e.g., Aniline, Benzene, etc.",
                key="new_molecule_input"
            )
            
            molecule_type = st.selectbox(
                "Molecule Type",
                ["Raw Material", "Intermediate", "Target Product", "By-Product"],
                key="molecule_type_select"
            )
            
            # Color mapping for different types
            color_map = {
                "Raw Material": "#3b82f6",      # Blue
                "Intermediate": "#f59e0b",      # Amber
                "Target Product": "#ef4444",    # Red
                "By-Product": "#8b5cf6",        # Purple
            }
            
            if st.button("➕ Add Molecule", disabled=not new_molecule_name):
                node_id = f"node_{st.session_state.node_counter}"
                st.session_state.node_counter += 1
                
                new_node = {
                    'id': node_id,
                    'label': new_molecule_name,
                    'type': molecule_type,
                    'color': color_map[molecule_type],
                    'size': 30 if molecule_type == "Target Product" else 25,
                    'font': {'size': 14, 'color': '#ffffff'}
                }
                st.session_state.value_chain_nodes.append(new_node)
                st.success(f"Added {new_molecule_name} as {molecule_type}")
                st.rerun()
        
        # Upload downstream applications
        with st.expander("📤 Upload Downstream Applications", expanded=False):
            st.markdown("""
            **Excel Format Expected:**
            - Category name in first column
            - Empty row (separator)
            - Category name in first column
            - Chemical names in rows below
            - Empty row (separator)
            - Next category name...
            
            The application categories will be automatically detected and visualized!
            """)
            
            applications_file = st.file_uploader(
                "Select Excel file (.xlsx, .xls)",
                type=['xlsx', 'xls'],
                key="applications_uploader"
            )
            
            if applications_file:
                try:
                    # Read Excel file
                    if applications_file.name.endswith('.xls'):
                        apps_df = pd.read_excel(applications_file, engine='xlrd', header=None)
                    else:
                        apps_df = pd.read_excel(applications_file, engine='openpyxl', header=None)
                    
                    # Extract categories and their chemicals
                    # Logic: After every empty row, first non-empty row is category name
                    # Subsequent rows until next empty row are chemicals
                    categories_dict = {}
                    current_category = None
                    prev_row_empty = True  # Start as if we just saw an empty row (for first category)
                    
                    for idx, row in apps_df.iterrows():
                        # Check if entire row is empty
                        row_is_empty = all(pd.isna(val) or str(val).strip() == '' for val in row)
                        
                        if row_is_empty:
                            prev_row_empty = True
                            continue
                        
                        first_col = row[0]
                        if pd.isna(first_col) or str(first_col).strip() == '':
                            continue
                        
                        first_col_str = str(first_col).strip()
                        
                        # Skip header row
                        if first_col_str.upper() in ['CHEMICAL_NAME', 'NAME', 'MOLECULE', 'SYNONYM', 'CAS_NO', 'MOLECULAR_FORMULA']:
                            prev_row_empty = False
                            continue
                        
                        # If previous row was empty, this is a category name
                        if prev_row_empty:
                            current_category = first_col_str
                            if current_category not in categories_dict:
                                categories_dict[current_category] = []
                            prev_row_empty = False
                        else:
                            # This is a chemical under current category
                            if current_category:
                                categories_dict[current_category].append(first_col_str)
                    
                    # Remove empty categories
                    categories_dict = {k: v for k, v in categories_dict.items() if v}
                    
                    if categories_dict:
                        st.success(f"✅ Found {len(categories_dict)} application categories")
                        
                        # Display found categories with counts in a scrollable container
                        st.write("**Detected Categories:**")
                        
                        # Create a summary view
                        category_summary = []
                        for cat, chemicals in categories_dict.items():
                            category_summary.append(f"• **{cat}**: {len(chemicals)} chemicals")
                        
                        st.markdown("\n".join(category_summary))
                        
                        # Predefined colors for different industry categories
                        category_color_mapping = {
                            'pharmaceutical': '#fbbf24',    # Yellow
                            'agrochemical': '#10b981',      # Green
                            'rubber': '#ec4899',            # Pink
                            'plastic': '#8b5cf6',           # Purple
                            'specialty': '#06b6d4',         # Cyan
                            'speciality': '#06b6d4',        # Cyan (alternative spelling)
                            'cosmetic': '#f97316',          # Orange
                            'veterinary': '#84cc16',        # Lime
                            'food': '#14b8a6',              # Teal
                            'intermediate': '#6366f1',      # Indigo
                            'colorant': '#ef4444',          # Red
                            'dye': '#d946ef',               # Fuchsia
                            'photochemical': '#0ea5e9',     # Sky blue
                        }
                        
                        # Assign colors to categories
                        default_colors = ['#10b981', '#fbbf24', '#ec4899', '#8b5cf6', '#06b6d4', 
                                        '#f97316', '#84cc16', '#14b8a6', '#6366f1', '#ef4444', '#d946ef', '#0ea5e9']
                        
                        category_colors = {}
                        color_idx = 0
                        
                        for cat in categories_dict.keys():
                            cat_lower = cat.lower()
                            color_assigned = False
                            
                            # Try to match with predefined colors
                            for keyword, color in category_color_mapping.items():
                                if keyword in cat_lower:
                                    category_colors[cat] = color
                                    color_assigned = True
                                    break
                            
                            # If no match, use default colors
                            if not color_assigned:
                                category_colors[cat] = default_colors[color_idx % len(default_colors)]
                                color_idx += 1
                        
                        if st.button("🔗 Create Value Chain with Categories"):
                            # Find target product nodes
                            target_products = [node for node in st.session_state.value_chain_nodes 
                                             if node['type'] == 'Target Product']
                            
                            if not target_products:
                                st.error("❌ No Target Product found! Please add a Target Product first.")
                            else:
                                target_node = target_products[0]
                                target_id = target_node['id']
                                
                                # Remove old group nodes and their edges
                                st.session_state.value_chain_nodes = [
                                    node for node in st.session_state.value_chain_nodes 
                                    if not node.get('is_group', False)
                                ]
                                st.session_state.value_chain_edges = [
                                    edge for edge in st.session_state.value_chain_edges
                                    if not any(node.get('is_group', False) 
                                             for node in st.session_state.value_chain_nodes 
                                             if node['id'] in [edge['from'], edge['to']])
                                ]
                                
                                # Create group nodes for each category
                                for category, chemicals in categories_dict.items():
                                    cat_id = f"group_{category.replace(' ', '_').replace('/', '_').lower()}"
                                    
                                    # Create label with count
                                    group_label = f"{category}\n({len(chemicals)} chemicals)"
                                    
                                    # Create tooltip with chemical names
                                    tooltip_chemicals = "\n".join(chemicals[:10])  # Show first 10
                                    if len(chemicals) > 10:
                                        tooltip_chemicals += f"\n... +{len(chemicals) - 10} more"
                                    
                                    # Create group node
                                    group_node = {
                                        'id': cat_id,
                                        'label': group_label,
                                        'type': 'Application Group',
                                        'color': category_colors[category],
                                        'size': 35,
                                        'font': {'size': 12, 'color': '#ffffff'},
                                        'is_group': True,
                                        'shape': 'box',
                                        'title': f"{category}:\n{tooltip_chemicals}"
                                    }
                                    st.session_state.value_chain_nodes.append(group_node)
                                    
                                    # Create edge from target to group
                                    new_edge = {
                                        'from': target_id,
                                        'to': cat_id,
                                        'width': 3,
                                        'color': category_colors[category],
                                        'dashes': False,
                                        'arrows': {'to': {'enabled': True, 'scaleFactor': 1.2}}
                                    }
                                    st.session_state.value_chain_edges.append(new_edge)
                                
                                st.success(f"✅ Created value chain with {len(categories_dict)} application categories connected to {target_node['label']}")
                                st.rerun()
                    else:
                        st.warning("⚠️ No categories found. Make sure your Excel follows the expected format.")
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    st.info("💡 **Tip:** Make sure your Excel file has category names followed by chemical names, separated by empty rows")
        
        # Show detected categories details (outside expander)
        if 'categories_dict' in locals() and categories_dict:
            st.markdown("---")
            st.markdown("#### 📋 Category Details")
            
            # Allow users to select a category to view details
            selected_category = st.selectbox(
                "Select category to view chemicals:",
                options=list(categories_dict.keys()),
                key="category_detail_select"
            )
            
            if selected_category:
                chemicals = categories_dict[selected_category]
                st.write(f"**{selected_category}** ({len(chemicals)} chemicals):")
                
                # Show chemicals in a scrollable text area
                chemicals_text = "\n".join([f"{i+1}. {chem}" for i, chem in enumerate(chemicals)])
                st.text_area(
                    "Chemicals list:",
                    value=chemicals_text,
                    height=200,
                    key="chemicals_display"
                )
        
        # Edit existing nodes
        if st.session_state.value_chain_nodes:
            with st.expander("✏️ Edit Molecules"):
                # Filter out group nodes from editing
                editable_nodes = [node for node in st.session_state.value_chain_nodes 
                                 if not node.get('is_group', False)]
                
                if editable_nodes:
                    node_to_edit = st.selectbox(
                        "Select Node to Edit",
                        options=[node['label'] for node in editable_nodes],
                        key="edit_node_select"
                    )
                    
                    if node_to_edit:
                        # Find the node
                        node_idx = next(i for i, node in enumerate(st.session_state.value_chain_nodes) 
                                      if node['label'] == node_to_edit and not node.get('is_group', False))
                        current_node = st.session_state.value_chain_nodes[node_idx]
                        
                        # Edit fields
                        edited_name = st.text_input(
                            "Edit Name",
                            value=current_node['label'],
                            key=f"edit_name_{node_idx}"
                        )
                        
                        color_map = {
                            "Raw Material": "#3b82f6",
                            "Intermediate": "#f59e0b",
                            "Target Product": "#ef4444",
                            "By-Product": "#8b5cf6",
                        }
                        
                        edited_type = st.selectbox(
                            "Edit Type",
                            ["Raw Material", "Intermediate", "Target Product", "By-Product"],
                            index=["Raw Material", "Intermediate", "Target Product", "By-Product"].index(current_node['type']),
                            key=f"edit_type_{node_idx}"
                        )
                        
                        col_update, col_delete = st.columns(2)
                        with col_update:
                            if st.button("💾 Update", key=f"update_{node_idx}"):
                                st.session_state.value_chain_nodes[node_idx]['label'] = edited_name
                                st.session_state.value_chain_nodes[node_idx]['type'] = edited_type
                                st.session_state.value_chain_nodes[node_idx]['color'] = color_map[edited_type]
                                st.session_state.value_chain_nodes[node_idx]['size'] = 30 if edited_type == "Target Product" else 25
                                st.success("✅ Updated!")
                                st.rerun()
                        
                        with col_delete:
                            if st.button("🗑️ Delete", key=f"delete_{node_idx}"):
                                # Remove node and its edges
                                node_id = current_node['id']
                                st.session_state.value_chain_nodes.pop(node_idx)
                                st.session_state.value_chain_edges = [
                                    edge for edge in st.session_state.value_chain_edges 
                                    if edge['from'] != node_id and edge['to'] != node_id
                                ]
                                st.success("✅ Deleted!")
                                st.rerun()
                else:
                    st.info("No editable molecules. Add molecules first.")
        
        # Add connections - simplified with only arrow direction
        if len([n for n in st.session_state.value_chain_nodes if not n.get('is_group', False)]) >= 2:
            with st.expander("🔗 Add Connections"):
                # Only show non-group nodes for manual connections
                connectable_nodes = [node['label'] for node in st.session_state.value_chain_nodes 
                                    if not node.get('is_group', False)]
                
                col_from, col_to = st.columns(2)
                
                with col_from:
                    from_molecule = st.selectbox(
                        "From (Upstream)",
                        options=connectable_nodes,
                        key="from_molecule"
                    )
                
                with col_to:
                    to_molecule = st.selectbox(
                        "To (Downstream)",
                        options=connectable_nodes,
                        key="to_molecule"
                    )
                
                if st.button("🔗 Add Connection"):
                    if from_molecule != to_molecule:
                        # Find node IDs
                        from_id = next(node['id'] for node in st.session_state.value_chain_nodes 
                                     if node['label'] == from_molecule and not node.get('is_group', False))
                        to_id = next(node['id'] for node in st.session_state.value_chain_nodes 
                                   if node['label'] == to_molecule and not node.get('is_group', False))
                        
                        # Check if edge already exists
                        edge_exists = any(
                            edge['from'] == from_id and edge['to'] == to_id
                            for edge in st.session_state.value_chain_edges
                        )
                        
                        if not edge_exists:
                            # Create simple arrow connection
                            new_edge = {
                                'from': from_id,
                                'to': to_id,
                                'width': 2,
                                'color': '#4ade80',
                                'dashes': False,
                                'arrows': {'to': {'enabled': True, 'scaleFactor': 1}}
                            }
                            st.session_state.value_chain_edges.append(new_edge)
                            st.success(f"✅ Connected {from_molecule} → {to_molecule}")
                            st.rerun()
                        else:
                            st.warning("⚠️ Connection already exists!")
                    else:
                        st.error("❌ Cannot connect a molecule to itself!")
        
        # Manage connections
        if st.session_state.value_chain_edges:
            with st.expander("🔧 Manage Connections"):
                for idx, edge in enumerate(st.session_state.value_chain_edges):
                    # Find node labels
                    from_node = next((node for node in st.session_state.value_chain_nodes 
                                    if node['id'] == edge['from']), None)
                    to_node = next((node for node in st.session_state.value_chain_nodes 
                                  if node['id'] == edge['to']), None)
                    
                    if from_node and to_node:
                        from_label = from_node['label']
                        to_label = to_node['label']
                        
                        col_edge, col_del = st.columns([3, 1])
                        with col_edge:
                            st.write(f"**{from_label}** → **{to_label}**")
                        with col_del:
                            if st.button("🗑️", key=f"del_edge_{idx}"):
                                st.session_state.value_chain_edges.pop(idx)
                                st.rerun()
        
        # Clear all
        st.markdown("---")
        if st.button("🗑️ Clear All", type="secondary"):
            st.session_state.value_chain_nodes = []
            st.session_state.value_chain_edges = []
            st.session_state.node_counter = 0
            st.rerun()
    
    with col_viz:
        st.markdown("#### 🔬 Value Chain Visualization")
        
        if st.session_state.value_chain_nodes:
            # Create nodes and edges for agraph
            nodes = []
            edges = []
            
            for node in st.session_state.value_chain_nodes:
                nodes.append(Node(
                    id=node['id'],
                    label=node['label'],
                    size=node['size'],
                    color=node['color'],
                    font=node.get('font', {'size': 14, 'color': '#ffffff'}),
                    title=node.get('title', f"{node['label']} ({node['type']})"),
                    shape=node.get('shape', 'box'),
                    borderWidth=2,
                    borderWidthSelected=4,
                    chosen=True,
                    physics=True
                ))
            
            for edge in st.session_state.value_chain_edges:
                edges.append(Edge(
                    source=edge['from'],
                    target=edge['to'],
                    width=edge.get('width', 2),
                    color=edge.get('color', '#4ade80'),
                    dashes=edge.get('dashes', False),
                    arrows=edge.get('arrows', {'to': {'enabled': True}})
                ))
            
            # Configure the graph
            config = Config(
                width=800,
                height=600,
                directed=True,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=False,
                node={'labelProperty': 'label'},
                link={'labelProperty': 'label', 'renderLabel': False},
                interaction={
                    'dragNodes': True,
                    'dragView': True,
                    'zoomView': True,
                    'hover': True,
                    'navigationButtons': True,
                    'keyboard': True
                },
                manipulation={
                    'enabled': False
                }
            )
            
            # Display the graph
            return_value = agraph(nodes=nodes, edges=edges, config=config)
            
            # Legend
            st.markdown("---")
            st.markdown("##### 🎨 Legend")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("🔵 **Raw Material**")
            with col2:
                st.markdown("🟠 **Intermediate**")
            with col3:
                st.markdown("🔴 **Target Product**")
            with col4:
                st.markdown("🟣 **By-Product**")
            
            # Show application category colors
            group_nodes = [node for node in st.session_state.value_chain_nodes if node.get('is_group', False)]
            if group_nodes:
                st.markdown("**Application Categories:**")
                
                # Create columns dynamically based on number of groups
                num_groups = len(group_nodes)
                cols_per_row = 3
                
                for i in range(0, num_groups, cols_per_row):
                    cols = st.columns(min(cols_per_row, num_groups - i))
                    for j, col in enumerate(cols):
                        if i + j < num_groups:
                            node = group_nodes[i + j]
                            category_name = node['label'].split('\n')[0]  # Get category name without count
                            with col:
                                st.markdown(f"<div style='background-color: {node['color']}; padding: 8px; border-radius: 5px; text-align: center; color: white; font-weight: bold; font-size: 0.85rem; margin: 2px;'>{category_name}</div>", unsafe_allow_html=True)
            
            # Statistics
            st.markdown("##### 📊 Statistics")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            molecule_nodes = [n for n in st.session_state.value_chain_nodes if not n.get('is_group', False)]
            group_count = len(group_nodes)
            
            with col_stat1:
                st.metric("Molecules", len(molecule_nodes))
            with col_stat2:
                st.metric("Categories", group_count)
            with col_stat3:
                st.metric("Connections", len(st.session_state.value_chain_edges))
            with col_stat4:
                target_count = sum(1 for node in molecule_nodes if node.get('type') == "Target Product")
                st.metric("Target Products", target_count)
        else:
            st.info("👈 Start by adding molecules using the controls panel")
            
            # Sample value chain
            if st.button("📚 Load Sample Value Chain"):
                sample_nodes = [
                    {'id': 'node_0', 'label': 'Benzene', 'type': 'Raw Material', 'color': '#3b82f6', 'size': 25, 'font': {'size': 14, 'color': '#ffffff'}},
                    {'id': 'node_1', 'label': 'Nitrobenzene', 'type': 'Intermediate', 'color': '#f59e0b', 'size': 25, 'font': {'size': 14, 'color': '#ffffff'}},
                    {'id': 'node_2', 'label': 'Aniline', 'type': 'Target Product', 'color': '#ef4444', 'size': 30, 'font': {'size': 14, 'color': '#ffffff'}},
                ]
                
                sample_edges = [
                    {'from': 'node_0', 'to': 'node_1', 'width': 2, 'color': '#4ade80', 'dashes': False, 'arrows': {'to': {'enabled': True, 'scaleFactor': 1}}},
                    {'from': 'node_1', 'to': 'node_2', 'width': 2, 'color': '#4ade80', 'dashes': False, 'arrows': {'to': {'enabled': True, 'scaleFactor': 1}}},
                ]
                
                # Sample categories from your data
                sample_categories = {
                    'Pharmaceuticals': ['Isatin', 'Phenazopyridine', 'Salverine', 'Sufentanil'],
                    'Rubbers': ['N,N\'-Diphenylguanidine', 'N-Phenyl-1-naphthylamine'],
                    'Agrochemicals': ['Fenuron', 'Forchlorfenuron', 'Carbendazim']
                }
                
                sample_colors = {
                    'Pharmaceuticals': '#fbbf24',
                    'Rubbers': '#ec4899',
                    'Agrochemicals': '#10b981'
                }
                
                st.session_state.value_chain_nodes = sample_nodes
                st.session_state.value_chain_edges = sample_edges
                st.session_state.node_counter = 3
                
                # Create group nodes
                for cat, chemicals in sample_categories.items():
                    cat_id = f"group_{cat.lower()}"
                    group_label = f"{cat}\n({len(chemicals)} chemicals)"
                    
                    group_node = {
                        'id': cat_id,
                        'label': group_label,
                        'type': 'Application Group',
                        'color': sample_colors[cat],
                        'size': 35,
                        'font': {'size': 12, 'color': '#ffffff'},
                        'is_group': True,
                        'shape': 'box',
                        'title': f"{cat}:\n" + "\n".join(chemicals)
                    }
                    st.session_state.value_chain_nodes.append(group_node)
                    
                    # Connect to target product
                    new_edge = {
                        'from': 'node_2',
                        'to': cat_id,
                        'width': 3,
                        'color': sample_colors[cat],
                        'dashes': False,
                        'arrows': {'to': {'enabled': True, 'scaleFactor': 1.2}}
                    }
                    st.session_state.value_chain_edges.append(new_edge)
                
                st.rerun()

with tab4:
    st.markdown("""
    ### 📈 Analytics & Insights
    Analyze your cleaned datasets with comprehensive analytics and visualizations.
    """)
    
    
    # Check for saved datasets only for EXIM and Data Overview tabs
    datasets_available = bool(st.session_state.saved_datasets)
    
    


    if not st.session_state.saved_datasets:
        st.info("📊 No saved datasets available yet. Please process and save data in the **Data Processing** tab first.")
    else:
        st.success(f"✅ {len(st.session_state.saved_datasets)} dataset(s) available for analysis")
        
        # Create sub-tabs for different analytics views
        analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4, analytics_tab5 = st.tabs([
            "📊 EXIM Analysis", 
            "📈 Data Overview",
            "👥 Customer Intelligence",
            "🛒 Websites Hunting",
            "🏭 Lead Generations"
        ])
        
        
        with analytics_tab1:
            st.subheader("📊 EXIM Analysis Table")
            st.write("Comprehensive Import, Export, and Global trade analysis with financial year breakdowns")
            
            # Helper function to determine financial year (April-March)
            def get_financial_year(date_val):
                """Convert date to financial year (April-March)"""
                try:
                    if pd.isna(date_val):
                        return None
                    
                    if isinstance(date_val, str):
                        date_obj = pd.to_datetime(date_val, dayfirst=True, errors='coerce')
                    else:
                        date_obj = pd.to_datetime(date_val)
                    
                    if pd.isna(date_obj):
                        return None
                    
                    year = date_obj.year
                    month = date_obj.month
                    
                    if month >= 4:  # April onwards
                        return f"FY {year}-{str(year+1)[2:]}"
                    else:  # Jan-March
                        return f"FY {year-1}-{str(year)[2:]}"
                except:
                    return None
            
            # Helper function to compute qty_kg for each row (for Import/Export data)
            def compute_row_metrics(row, qty_col, unit_col, value_col):
                """
                Convert each row to qty_kg and price_usd_per_kg based on unit.
                Returns: (qty_kg, price_usd_per_kg) or (None, None) if invalid
                """
                try:
                    # Get values
                    qty = row[qty_col]
                    unit = row[unit_col]
                    unit_value_usd = row[value_col]
                    
                    # Validate
                    if pd.isna(qty) or pd.isna(unit) or pd.isna(unit_value_usd):
                        return None, None
                    
                    if qty <= 0 or unit_value_usd <= 0:
                        return None, None
                    
                    # Normalize unit string
                    unit_str = str(unit).strip().upper()
                    
                    # Process based on unit
                    if 'METRIC TON' in unit_str or unit_str == 'MTS' or unit_str == 'MT':
                        # Unit is METRIC TON
                        qty_kg = float(qty) * 1000  # 1 MT = 1000 kg
                        price_usd_per_kg = float(unit_value_usd) / 1000.0  # Convert $/MT to $/kg
                        return qty_kg, price_usd_per_kg
                    
                    elif 'KILOGRAM' in unit_str or unit_str == 'KGS' or unit_str == 'KG':
                        # Unit is KILOGRAMS
                        qty_kg = float(qty)
                        price_usd_per_kg = float(unit_value_usd)  # Already $/kg
                        return qty_kg, price_usd_per_kg
                    
                    else:
                        # Other units (GRAMS, NUMBER, etc.) - exclude
                        return None, None
                
                except:
                    return None, None
            
            # Helper function for Global data (aggregation-based calculation)
            def compute_global_metrics(row, qty_col, value_col):
                """
                Extract quantity (in kg) and total value (in USD) for Global data.
                Returns: (qty_kg, total_value_usd) or (None, None) if invalid
                """
                try:
                    # Get values
                    qty = row[qty_col]
                    total_value = row[value_col]
                    
                    # Validate
                    if pd.isna(qty) or pd.isna(total_value):
                        return None, None
                    
                    if qty <= 0 or total_value <= 0:
                        return None, None
                    
                    # Assume quantity is in kg (adjust if your data uses different units)
                    qty_kg = float(qty)
                    total_value_usd = float(total_value)
                    
                    return qty_kg, total_value_usd
                
                except:
                    return None, None
            
            # Process all saved datasets with CORRECT calculation logic
            def process_exim_data():
                """
                Process all saved datasets and create EXIM analysis data.
                - Import/Export: Uses ONLY METRIC TON and KILOGRAMS rows for weighted average price calculation.
                - Global: Uses aggregation (sum of values / sum of quantities) for Million$/MT calculation.
                """
                
                exim_data = {
                    'Imports': [],
                    'Exports': [],
                    'Global': []
                }
                
                financial_years = [
                    "FY 2020-21", "FY 2021-22", "FY 2022-23", 
                    "FY 2023-24", "FY 2024-25", "FY 2025-26"
                ]
                
                # Track excluded rows for logging
                excluded_stats = {
                    'total': 0, 
                    'wrong_unit': 0, 
                    'missing_data': 0,
                    'by_dataset': {}
                }
                
                for dataset_name, dataset_info in st.session_state.saved_datasets.items():
                    data_type = dataset_info['type']
                    df = dataset_info['data'].copy()
                    
                    # Determine section
                    if data_type == 'Import':
                        section = 'Imports'
                    elif data_type == 'Export':
                        section = 'Exports'
                    else:
                        section = 'Global'
                    
                    # Check for required columns
                    if 'Standardized Name' not in df.columns:
                        st.warning(f"⚠️ Dataset '{dataset_name}' missing 'Standardized Name' column. Skipping.")
                        continue
                    
                    # Find required columns
                    date_col = None
                    for col in df.columns:
                        if 'date' in col.lower() or 'period' in col.lower():
                            date_col = col
                            break
                    
                    qty_col = None
                    for col in df.columns:
                        if 'quantity' in col.lower() or 'qty' in col.lower() or 'weight' in col.lower():
                            qty_col = col
                            break
                    
                    unit_col = None
                    for col in df.columns:
                        if 'unit' in col.lower() or 'uqc' in col.lower():
                            unit_col = col
                            break
                    
                    # Find Value column - different logic for Global vs Import/Export
                    value_col = None
                    if section == 'Global':
                        # For Global: look for total value/amount columns
                        for col in df.columns:
                            col_lower = col.lower()
                            if 'value' in col_lower or 'amount' in col_lower or 'usd' in col_lower:
                                value_col = col
                                break
                    else:
                        # For Import/Export: look for Unit Value (USD) column
                        for col in df.columns:
                            col_lower = col.lower()
                            if 'unit value' in col_lower and 'usd' in col_lower:
                                value_col = col
                                break
                            elif 'unit price' in col_lower or 'price' in col_lower:
                                value_col = col
                                break
                        
                        if not value_col:
                            # Fallback to any value/amount column
                            for col in df.columns:
                                col_lower = col.lower()
                                if 'value' in col_lower or 'amount' in col_lower or 'fob' in col_lower or 'cif' in col_lower:
                                    value_col = col
                                    break
                    
                    # Check required columns based on section
                    if section == 'Global':
                        # Global doesn't need unit_col
                        if not all([date_col, qty_col, value_col]):
                            st.warning(f"⚠️ Dataset '{dataset_name}' missing required columns (Date, Quantity, Value). Skipping.")
                            st.info(f"Found: Date={date_col}, Qty={qty_col}, Value={value_col}")
                            continue
                    else:
                        if not all([date_col, qty_col, unit_col, value_col]):
                            st.warning(f"⚠️ Dataset '{dataset_name}' missing required columns (Date, Quantity, Unit, Value). Skipping.")
                            st.info(f"Found: Date={date_col}, Qty={qty_col}, Unit={unit_col}, Value={value_col}")
                            continue
                    
                    # Add financial year column
                    df['FY'] = df[date_col].apply(get_financial_year)
                    
                    # Compute metrics based on section type
                    if section == 'Global':
                        # Global: Extract qty_kg and total_value_usd
                        metrics = df.apply(lambda row: compute_global_metrics(row, qty_col, value_col), axis=1)
                        df['qty_kg'] = metrics.apply(lambda x: x[0] if x[0] is not None else None)
                        df['total_value_usd'] = metrics.apply(lambda x: x[1] if x[1] is not None else None)
                        df['price_usd_per_kg'] = None  # Not used for Global
                    else:
                        # Import/Export: Compute qty_kg and price_usd_per_kg for each row
                        metrics = df.apply(lambda row: compute_row_metrics(row, qty_col, unit_col, value_col), axis=1)
                        df['qty_kg'] = metrics.apply(lambda x: x[0] if x[0] is not None else None)
                        df['price_usd_per_kg'] = metrics.apply(lambda x: x[1] if x[1] is not None else None)
                        df['total_value_usd'] = None  # Not used for Import/Export
                    
                    # Count excluded rows
                    original_count = len(df)
                    
                    # Filter: Keep only rows with valid data
                    if section == 'Global':
                        df_valid = df[df['qty_kg'].notna() & df['total_value_usd'].notna()].copy()
                    else:
                        df_valid = df[df['qty_kg'].notna() & df['price_usd_per_kg'].notna()].copy()
                    
                    excluded_count = original_count - len(df_valid)
                    if excluded_count > 0:
                        excluded_stats['total'] += excluded_count
                        excluded_stats['by_dataset'][dataset_name] = {
                            'original': original_count,
                            'excluded': excluded_count,
                            'kept': len(df_valid),
                            'percentage_kept': (len(df_valid) / original_count * 100) if original_count > 0 else 0
                        }
                        if section != 'Global':
                            st.info(f"ℹ️ Dataset '{dataset_name}': Excluded {excluded_count} rows (non MT/KG units or missing data) | Kept: {len(df_valid)} rows ({(len(df_valid)/original_count*100):.1f}%)")
                        else:
                            st.info(f"ℹ️ Dataset '{dataset_name}' (Global): Excluded {excluded_count} rows (missing data) | Kept: {len(df_valid)} rows ({(len(df_valid)/original_count*100):.1f}%)")
                    
                    if len(df_valid) == 0:
                        st.warning(f"⚠️ Dataset '{dataset_name}': No valid rows after filtering. Skipping.")
                        continue
                    
                    # Get unique products
                    products = df_valid['Standardized Name'].unique()
                    
                    for product in products:
                        product_df = df_valid[df_valid['Standardized Name'] == product].copy()
                        
                        # Initialize row
                        row = {
                            'Product': f"{product} — [Dataset: {dataset_name}]",
                            'Product_Clean': product,
                            'Dataset': dataset_name,
                            'Relation to Product': st.session_state.product_relations.get(f"{product}_{dataset_name}", "Product")
                        }
                        
                        # Calculate metrics for each FY
                        for fy in financial_years:
                            fy_data = product_df[product_df['FY'] == fy].copy()
                            
                            if not fy_data.empty:
                                if section == 'Global':
                                    # GLOBAL CALCULATION: Aggregation-based
                                    # Step 1: Sum all trade values (USD) and convert to Million USD
                                    total_value_usd = fy_data['total_value_usd'].sum()
                                    total_value_million_usd = total_value_usd / 1_000_000.0
                                    
                                    # Step 2: Sum all quantities (kg) and convert to MT
                                    total_qty_kg = fy_data['qty_kg'].sum()
                                    total_qty_mt = total_qty_kg / 1000.0
                                    
                                    # Step 3: Calculate Million$/MT (divide Million USD by MT)
                                    if total_qty_mt > 0 and total_value_million_usd > 0:
                                        # Price = Million USD / MT = Million$/MT
                                        price_million_usd_per_mt = (total_value_million_usd / total_qty_mt)*1000
                                        price_per_mt = price_million_usd_per_mt * 1000
                                        
                                        # Store with proper rounding
                                        row[f'{fy}_Qty'] = round(total_qty_mt, 2)  # 2 decimal places for MT
                                        row[f'{fy}_Price'] = round(price_million_usd_per_mt, 4)  # 4 decimal places for Million$/MT
                                        
                                        # Store raw values for tooltip/debugging
                                        row[f'{fy}_TotalValue'] = total_value_usd
                                        row[f'{fy}_TotalValueMillionUSD'] = total_value_million_usd
                                        row[f'{fy}_TotalQtyMT'] = total_qty_mt
                                        row[f'{fy}_RowCount'] = len(fy_data)
                                    else:
                                        row[f'{fy}_Qty'] = None
                                        row[f'{fy}_Price'] = None
                                
                                else:
                                    # IMPORT/EXPORT CALCULATION: Weighted average
                                    total_qty_kg = fy_data['qty_kg'].sum()
                                    
                                    if total_qty_kg > 0:
                                        # Weighted average price calculation
                                        numerator = (fy_data['qty_kg'] * fy_data['price_usd_per_kg']).sum()
                                        denominator = total_qty_kg
                                        weighted_price_per_kg = numerator / denominator
                                        
                                        # Convert qty_kg to MT for display
                                        total_qty_mt = total_qty_kg / 1000.0
                                        
                                        # Store with proper rounding
                                        row[f'{fy}_Qty'] = round(total_qty_mt, 2)  # 2 decimal places for MT
                                        row[f'{fy}_Price'] = round(weighted_price_per_kg, 1)  # 1 decimal place for $/kg
                                        
                                        # Store raw values for tooltip/debugging
                                        row[f'{fy}_Numerator'] = numerator
                                        row[f'{fy}_Denominator'] = denominator
                                        row[f'{fy}_RowCount'] = len(fy_data)
                                    else:
                                        row[f'{fy}_Qty'] = None
                                        row[f'{fy}_Price'] = None
                            else:
                                row[f'{fy}_Qty'] = None
                                row[f'{fy}_Price'] = None
                        
                        exim_data[section].append(row)
                
                # Display exclusion statistics with details
                if excluded_stats['total'] > 0:
                    st.markdown("---")
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(251, 146, 60, 0.1) 100%); 
                                padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b; margin: 20px 0;'>
                        <h4 style='color: #fbbf24; margin: 0 0 10px 0; font-weight: 700;'>
                            ⚠️ Data Quality Report
                        </h4>
                        <p style='color: #ffffff; margin: 0 0 10px 0; font-weight: 600;'>
                            Some rows were excluded from analysis due to unsupported units or missing data.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_exc1, col_exc2 = st.columns(2)
                    with col_exc1:
                        st.metric("📊 Total Rows Excluded", f"{excluded_stats['total']:,}")
                    with col_exc2:
                        total_rows = sum(info['original'] for info in excluded_stats['by_dataset'].values())
                        kept_rows = sum(info['kept'] for info in excluded_stats['by_dataset'].values())
                        st.metric("✅ Total Rows Used", f"{kept_rows:,} ({(kept_rows/total_rows*100):.1f}%)")
                    
                    with st.expander("📋 View Detailed Exclusion Report by Dataset"):
                        exclusion_report = []
                        for ds_name, info in excluded_stats['by_dataset'].items():
                            exclusion_report.append({
                                'Dataset': ds_name,
                                'Original Rows': f"{info['original']:,}",
                                'Rows Used': f"{info['kept']:,}",
                                'Rows Excluded': f"{info['excluded']:,}",
                                'Success Rate': f"{info['percentage_kept']:.1f}%"
                            })
                        
                        if exclusion_report:
                            exc_df = pd.DataFrame(exclusion_report)
                            st.dataframe(exc_df, use_container_width=True)
                        
                        st.info("""
                        **Common reasons for exclusion:**
                        - **Import/Export**: Units other than METRIC TON or KILOGRAMS (e.g., GRAMS, NUMBER, PIECES)
                        - **Global**: Missing Quantity or Value data
                        - Zero or negative values
                        - Invalid date formats
                        """)
                
                return exim_data
            
            # Process data with detailed logging
            st.markdown("---")
            st.markdown("### 🔄 Processing Status")
            
            with st.spinner("🔄 Processing EXIM data from all saved datasets..."):
                exim_data = process_exim_data()
            
            st.success("✅ Data processing complete!")
            
            # Add unit test / validation section
            st.markdown("---")
            st.markdown("### 🧪 Calculation Validation")
            
            with st.expander("📊 View Sample Calculations (Unit Tests)"):
                # Import/Export Example
                st.markdown("#### Import/Export Calculation (Weighted Average)")
                st.markdown("""
                **Example Weighted Average Calculation:**
                
                Given data for "Aniline" in FY 2023-24 from 3 shipments:
                
                | Row | Quantity | Unit | Unit Value (USD) | qty_kg | price_usd_per_kg |
                |-----|----------|------|------------------|--------|------------------|
                | 1   | 500      | KILOGRAMS | $2.50/kg   | 500    | $2.50           |
                | 2   | 2        | METRIC TON | $2000/MT  | 2000   | $2.00           |
                | 3   | 1500     | KILOGRAMS | $3.00/kg   | 1500   | $3.00           |
                | 4*  | 50       | GRAMS     | $5.00/g    | ❌ EXCLUDED (wrong unit) | |
                
                *Row 4 is excluded because unit is GRAMS (not MT or KG)
                
                **Step-by-Step Calculation:**
                ```
                Step 1: Convert all to qty_kg and price_usd_per_kg
                  Row 1: qty_kg = 500, price = $2.50/kg
                  Row 2: qty_kg = 2×1000 = 2000, price = $2000/1000 = $2.00/kg
                  Row 3: qty_kg = 1500, price = $3.00/kg
                
                Step 2: Calculate weighted average
                  Numerator = Σ(qty_kg × price_usd_per_kg)
                            = (500 × 2.50) + (2000 × 2.00) + (1500 × 3.00)
                            = 1250 + 4000 + 4500
                            = 9750
                  
                  Denominator = Σ(qty_kg)
                              = 500 + 2000 + 1500
                              = 4000
                  
                  Weighted Avg Price = 9750 ÷ 4000 = 2.4375 $/kg
                
                Step 3: Round to 1 decimal
                  Price ($/kg) = 2.4 $/kg
                
                Step 4: Convert total quantity to MT
                  Total Qty (MT) = 4000 ÷ 1000 = 4.00 MT
                
                Final Display: Aniline | FY 2023-24 | 4.00 MT @ $2.4/kg
                ```
                """)
                
                st.markdown("---")
                
                # Global Example
                st.markdown("#### Global Trade Calculation (Aggregation-Based)")
                st.markdown("""
                **Example Global Trade Calculation:**
                
                Given country-level data for "Benzene" in FY 2023-24 from 4 countries:
                
                | Row | Country | Quantity (kg) | Total Value (USD) |
                |-----|---------|---------------|-------------------|
                | 1   | USA     | 500,000       | $1,250,000        |
                | 2   | China   | 2,000,000     | $4,200,000        |
                | 3   | Germany | 1,500,000     | $3,150,000        |
                | 4   | Japan   | 800,000       | $1,680,000        |
                
                **Step-by-Step Calculation:**
                ```
                Step 1: Sum all trade values (USD) and convert to Million USD
                  Total Value (USD) = $1,250,000 + $4,200,000 + $3,150,000 + $1,680,000
                                    = $10,280,000 USD
                  
                  Total Value (Million USD) = $10,280,000 ÷ 1,000,000
                                            = $10.28 Million USD
                
                Step 2: Sum all quantities (kg) and convert to MT
                  Total Qty (kg) = 500,000 + 2,000,000 + 1,500,000 + 800,000
                                 = 4,800,000 kg
                  
                  Total Qty (MT) = 4,800,000 ÷ 1,000
                                 = 4,800 MT
                
                Step 3: Calculate Million$/MT (divide Million USD by MT)
                  Price (Million$/MT) = Total Value (Million USD) ÷ Total Qty (MT)
                                      = $10.28 Million ÷ 4,800 MT
                                      = $0.002142 Million per MT
                  
                  Rounded = $0.0021 Million/MT (4 decimal places)
                
                Final Display: Benzene | FY 2023-24 | 4,800.00 MT @ $0.0021M/MT
                
                Interpretation: Each metric ton costs $0.0021 million = $2,100
                ```
                
                **Key Differences from Import/Export:**
                - ✅ NO weighted average - simple aggregation
                - ✅ Convert USD to Million USD FIRST
                - ✅ Price displayed as Million$/MT (not $/kg or $/MT)
                - ✅ Uses sum of total values, not unit values
                - ✅ No unit column required
                - ✅ Formula: (Sum USD ÷ 1M) ÷ (Sum kg ÷ 1000)
                """)
                
                # Visual comparison
                st.markdown("---")
                st.markdown("#### Visual Comparison")
                st.markdown("""
                **Import/Export Flow:**
                ```
                Raw Data → Filter Units → Convert to kg → Weighted Avg → $/kg
                   ↓           ↓              ↓               ↓            ↓
                Multiple    Keep only    All in kg &    Σ(qty×price)   Display
                 rows       MT & KG      same unit      ─────────      per kg
                                                         Σ(qty)
                ```
                
                **Global Trade Flow:**
                ```
                Country Data → Sum Values → Convert to M$ → Sum Quantities → Convert to MT → Divide → M$/MT
                   ↓              ↓              ↓              ↓                ↓           ↓         ↓
                Multiple      Total USD    USD÷1,000,000   Total kg         kg÷1,000    M$÷MT   Display
                countries                   = Million$                        = MT                 M$/MT
                ```
                """)
            
            st.markdown("---")
            financial_years = [
                "FY 2020-21", "FY 2021-22", "FY 2022-23", 
                "FY 2023-24", "FY 2024-25", "FY 2025-26"
            ]
            
            relation_options = [
                "Product", "Downstream", "Raw Material (RM)", 
                "Completion Product", "N+1", "N+2", "N-1", "N-2"
            ]
            
            # Section colors
            section_colors = {
                'Imports': '#3b82f6',    # Blue
                'Exports': '#f59e0b',    # Amber
                'Global': '#10b981'      # Green
            }
            
            section_icons = {
                'Imports': '📥',
                'Exports': '📤',
                'Global': '🌍'
            }
            
            for section in ['Imports', 'Exports', 'Global']:
                # Section header with color
                st.markdown(f"""
                <div style='background: linear-gradient(90deg, {section_colors[section]}22 0%, {section_colors[section]}11 100%); 
                            padding: 15px; border-radius: 10px; border-left: 5px solid {section_colors[section]}; margin: 20px 0;'>
                    <h3 style='color: {section_colors[section]}; margin: 0; font-weight: 700;'>
                        {section_icons[section]} {section}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                if not exim_data[section]:
                    st.info(f"No {section.lower()} data available. Save datasets in Tab 1 to see analysis.")
                    continue
                
                # Create display dataframe
                display_data = []
                
                for idx, row in enumerate(exim_data[section]):
                    display_row = {
                        'Product': row['Product'],
                        'Relation': row['Relation to Product']
                    }
                    
                    # Add FY columns with better formatting
                    for fy in financial_years:
                        qty = row.get(f'{fy}_Qty')
                        price = row.get(f'{fy}_Price')
                        
                        # Format quantity
                        if qty is not None and qty > 0:
                            display_row[f'{fy}\nQty (MT)'] = f"{qty:,.2f}"
                        else:
                            display_row[f'{fy}\nQty (MT)'] = "—"
                        
                        # Format price - label changes for Global
                        price_label = f'{fy}\nPrice (M$/MT)' if section == 'Global' else f'{fy}\nPrice ($/kg)'
                        if price is not None and price > 0:
                            if section == 'Global':
                                display_row[price_label] = f"${price:.4f}M"
                            else:
                                display_row[price_label] = f"${price:.1f}"
                        else:
                            display_row[price_label] = "—"
                    
                    display_data.append(display_row)
                
                # Create DataFrame
                if display_data:
                    df_display = pd.DataFrame(display_data)
                    
                    st.write(f"**Total Products: {len(display_data)}**")
                    
                    # Display styled table
                    st.dataframe(
                        df_display, 
                        use_container_width=True,
                        height=min(600, len(display_data) * 40 + 50)
                    )
                    
                    # Show calculation details in expander
                    with st.expander(f"🔍 View Calculation Details for {section}"):
                        if section == 'Global':
                            st.write("**Global Trade Aggregation Calculation:**")
                            st.write("Formula: `Price (Million$/MT) = Total Value (Million USD) ÷ Total Quantity (MT)`")
                            st.write("Sum all trade values (convert to Million USD), sum all quantities (convert to MT), then divide.")
                        else:
                            st.write("**Weighted Average Price Calculation:**")
                            st.write("Formula: `Price ($/kg) = Σ(qty_kg × price_usd_per_kg) / Σ(qty_kg)`")
                            st.write("Only rows with units **METRIC TON** or **KILOGRAMS** are included.")
                        
                        # Show detailed breakdown for selected products
                        for idx, row_data in enumerate(exim_data[section][:5]):  # Show first 5 products
                            st.markdown(f"**{row_data['Product']}:**")
                            
                            detail_data = []
                            for fy in financial_years:
                                if row_data.get(f'{fy}_Qty'):
                                    if section == 'Global':
                                        total_value = row_data.get(f'{fy}_TotalValue', 0)
                                        total_value_m = row_data.get(f'{fy}_TotalValueMillionUSD', 0)
                                        total_qty_mt = row_data.get(f'{fy}_TotalQtyMT', 0)
                                        row_count = row_data.get(f'{fy}_RowCount', 0)
                                        
                                        detail_data.append({
                                            'FY': fy,
                                            'Total Value': f"${total_value:,.0f}",
                                            'Value (Million USD)': f"${total_value_m:.3f}M",
                                            'Total Qty (MT)': f"{total_qty_mt:,.2f}",
                                            'Price (M$/MT)': f"${row_data[f'{fy}_Price']:.4f}M",
                                            'Rows Used': row_count
                                        })
                                    else:
                                        numerator = row_data.get(f'{fy}_Numerator', 0)
                                        denominator = row_data.get(f'{fy}_Denominator', 0)
                                        row_count = row_data.get(f'{fy}_RowCount', 0)
                                        
                                        detail_data.append({
                                            'FY': fy,
                                            'Total Qty (kg)': f"{denominator:,.2f}",
                                            'Weighted Sum': f"{numerator:,.2f}",
                                            'Price ($/kg)': f"${row_data[f'{fy}_Price']:.1f}",
                                            'Rows Used': row_count
                                        })
                            
                            if detail_data:
                                detail_df = pd.DataFrame(detail_data)
                                st.dataframe(detail_df, use_container_width=True)
                            else:
                                st.write("No data available for this product")
                            
                            if idx < 4:  # Don't add separator after last item
                                st.markdown("---")
                    
                    # Edit Relations Section
                    with st.expander(f"✏️ Edit Product Relations for {section}"):
                        st.write("Update the relationship classification for each product:")
                        
                        # Create form for editing relations
                        for idx, row_data in enumerate(exim_data[section]):
                            product_key = f"{row_data['Product_Clean']}_{row_data['Dataset']}"
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{row_data['Product']}**")
                            with col2:
                                current_relation = st.session_state.product_relations.get(product_key, "Product")
                                new_relation = st.selectbox(
                                    "Relation",
                                    options=relation_options,
                                    index=relation_options.index(current_relation) if current_relation in relation_options else 0,
                                    key=f"relation_{section}_{idx}"
                                )
                                
                                # Update session state
                                st.session_state.product_relations[product_key] = new_relation
                        
                        if st.button(f"💾 Save Changes for {section}", key=f"save_relations_{section}"):
                            st.success(f"✅ Relations updated for {section}!")
                            st.rerun()
                    
                    # Summary statistics with enhanced visuals
                    st.markdown("---")
                    st.markdown("##### 📊 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_products = len(display_data)
                        st.metric("Total Products", total_products)
                    with col2:
                        latest_fy = financial_years[-1]
                        # Calculate total quantity for products with valid data
                        total_qty = sum([
                            row.get(f'{latest_fy}_Qty', 0) 
                            for row in exim_data[section] 
                            if row.get(f'{latest_fy}_Qty') is not None
                        ])
                        st.metric(f"{latest_fy} Qty", f"{total_qty:,.2f} MT")
                    with col3:
                        # Calculate weighted average price for latest FY
                        valid_products = [
                            row for row in exim_data[section] 
                            if row.get(f'{latest_fy}_Price') is not None and row.get(f'{latest_fy}_Qty') is not None
                        ]
                        
                        if valid_products:
                            if section == 'Global':
                                # For Global: calculate overall Million$/MT from total values
                                total_value_sum = sum([
                                    row.get(f'{latest_fy}_TotalValue', 0) 
                                    for row in valid_products
                                ])
                                total_qty_sum = sum([
                                    row.get(f'{latest_fy}_TotalQtyMT', 0) 
                                    for row in valid_products
                                ])
                                
                                if total_qty_sum > 0:
                                    total_value_million = total_value_sum / 1_000_000.0
                                    avg_price = total_value_million / total_qty_sum
                                    st.metric(f"{latest_fy} Avg Price", f"${avg_price:.4f}M/MT")
                                else:
                                    st.metric(f"{latest_fy} Avg Price", "N/A")
                            else:
                                # For Import/Export: use weighted average
                                total_numerator = sum([
                                    row.get(f'{latest_fy}_Numerator', 0) 
                                    for row in valid_products
                                ])
                                total_denominator = sum([
                                    row.get(f'{latest_fy}_Denominator', 0) 
                                    for row in valid_products
                                ])
                                
                                if total_denominator > 0:
                                    avg_price = total_numerator / total_denominator
                                    st.metric(f"{latest_fy} Avg Price", f"${avg_price:.1f}/kg")
                                else:
                                    st.metric(f"{latest_fy} Avg Price", "N/A")
                        else:
                            st.metric(f"{latest_fy} Avg Price", "N/A")
                    with col4:
                        # Total value in latest FY
                        if valid_products:
                            if section == 'Global':
                                total_value_usd = sum([
                                    row.get(f'{latest_fy}_TotalValue', 0) 
                                    for row in valid_products
                                ])
                            else:
                                total_value_usd = sum([
                                    row.get(f'{latest_fy}_Numerator', 0) 
                                    for row in valid_products
                                ])
                            
                            if total_value_usd > 0:
                                st.metric(f"{latest_fy} Total Value", f"${total_value_usd/1_000_000:,.1f}M")
                            else:
                                st.metric(f"{latest_fy} Total Value", "N/A")
                        else:
                            st.metric(f"{latest_fy} Total Value", "N/A")
                    
                    # Download section data with enhanced formatting
                    st.markdown("---")
                    st.markdown("##### 📥 Export Data")
                    
                    # Create export dataframe with numeric values
                    export_data = []
                    for row in exim_data[section]:
                        export_row = {
                            'Product': row['Product'],
                            'Relation to Product': row['Relation to Product']
                        }
                        for fy in financial_years:
                            qty = row.get(f'{fy}_Qty')
                            price = row.get(f'{fy}_Price')
                            
                            export_row[f'{fy} Qty (MT)'] = qty if qty is not None else ''
                            
                            # Label price column based on section
                            price_label = f'{fy} Price (M$/MT)' if section == 'Global' else f'{fy} Price ($/kg)'
                            export_row[price_label] = price if price is not None else ''
                        export_data.append(export_row)
                    
                    df_export = pd.DataFrame(export_data)
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_export.to_excel(writer, index=False, sheet_name=section)
                    output.seek(0)
                    
                    st.download_button(
                        label=f"📥 Download {section} Data",
                        data=output,
                        file_name=f"exim_{section.lower()}_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_{section}"
                    )
                
                st.markdown("---")
            
            # Export all EXIM data with enhanced UI
            if any(exim_data.values()):
                st.markdown("---")
                st.markdown("""
                <div style='background: linear-gradient(90deg, #10b98122 0%, #3b82f611 100%); 
                            padding: 20px; border-radius: 12px; border: 2px solid #10b981; margin: 20px 0;'>
                    <h4 style='color: #10b981; margin: 0 0 10px 0; font-weight: 700;'>
                        📦 Export Complete EXIM Analysis
                    </h4>
                    <p style='color: #ffffff; margin: 0; font-weight: 600;'>
                        Download a comprehensive report with all Import, Export, and Global trade data in a single Excel file
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Summary of what will be exported
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                with col_summary1:
                    st.metric("Imports Products", len(exim_data['Imports']))
                with col_summary2:
                    st.metric("Exports Products", len(exim_data['Exports']))
                with col_summary3:
                    st.metric("Global Products", len(exim_data['Global']))
                
                if st.button("📥 Download Complete EXIM Report", key="download_complete_exim", type="primary"):
                    with st.spinner("📦 Generating comprehensive EXIM report..."):
                        output_all = BytesIO()
                        with pd.ExcelWriter(output_all, engine='openpyxl') as writer:
                            for section in ['Imports', 'Exports', 'Global']:
                                if exim_data[section]:
                                    export_data = []
                                    for row in exim_data[section]:
                                        export_row = {
                                            'Product': row['Product'],
                                            'Relation to Product': row['Relation to Product']
                                        }
                                        for fy in financial_years:
                                            qty = row.get(f'{fy}_Qty')
                                            price = row.get(f'{fy}_Price')
                                            
                                            export_row[f'{fy} Qty (MT)'] = qty if qty is not None else ''
                                            
                                            # Label price column based on section
                                            price_label = f'{fy} Price (M$/MT)' if section == 'Global' else f'{fy} Price ($/kg)'
                                            export_row[price_label] = price if price is not None else ''
                                        export_data.append(export_row)
                                    
                                    df_section = pd.DataFrame(export_data)
                                    df_section.to_excel(writer, index=False, sheet_name=section)
                        
                        output_all.seek(0)
                        


        with analytics_tab2:
            st.subheader("📊 Data Overview & Insights")
            st.write("Interactive visualizations with quarterly trends, geographic distribution, and supply chain mapping")
            
            # Dataset selection
            st.markdown("---")
            col_select1, col_select2, col_select3 = st.columns([2, 1, 1])
            
            with col_select1:
                selected_dataset = st.selectbox(
                    "📁 Select Dataset for Analysis:",
                    options=list(st.session_state.saved_datasets.keys()),
                    key="overview_dataset_select"
                )
            
            dataset_info = st.session_state.saved_datasets[selected_dataset]
            
            with col_select2:
                st.metric("Data Type", dataset_info['type'])
            with col_select3:
                st.metric("Total Rows", f"{dataset_info['rows']:,}")
            
            # Process dataset
            st.markdown("---")
            with st.spinner("🔄 Processing dataset for analytics..."):
                df_processed, error = process_dataset_for_analytics(dataset_info['data'])
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 Ensure your dataset has Date, Quantity, Unit, and Unit Value columns with MT/KG entries.")
            else:
                original_rows = len(dataset_info['data'])
                processed_rows = len(df_processed)
                excluded_rows = original_rows - processed_rows
                
                st.success(f"✅ Processed {processed_rows:,} rows ({excluded_rows:,} excluded - non MT/KG units)")
                
                # Add Financial Year column
                def get_fy_from_quarter(quarter):
                    """Extract FY from quarter string"""
                    if pd.isna(quarter):
                        return None
                    try:
                        return quarter.split()[0]
                    except:
                        return None
                
                df_processed['FY'] = df_processed['Quarter'].apply(get_fy_from_quarter)
                
                # Key metrics
                st.markdown("### 📊 Key Metrics")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                total_volume = df_processed['Quantity_MT'].sum()
                avg_price = (df_processed['Quantity_MT'] * df_processed['Price_USD_per_kg']).sum() / total_volume
                unique_quarters = df_processed['Quarter'].nunique()
                
                with col_m1:
                    st.metric("Total Volume", f"{total_volume:,.2f} MT")
                if dataset_info['type'] != 'Global':
                   with col_m2:
                        st.metric("Weighted Avg Price", f"${avg_price:.1f}/kg")
                with col_m3:
                    st.metric("Time Periods", f"{unique_quarters} Quarters")
                with col_m4:
                    if 'Standardized Name' in df_processed.columns:
                        st.metric("Unique Products", df_processed['Standardized Name'].nunique())
                    else:
                        st.metric("Data Points", f"{processed_rows:,}")
                
                # 1. Quarterly Volume & Price Analysis
                st.markdown("---")
                st.markdown("### 📈 Quarterly Volume & Price Analysis")
                
                # Group by quarter
                quarterly = df_processed.groupby('Quarter').agg({
                    'Quantity_MT': 'sum',
                    'Price_USD_per_kg': lambda x: (df_processed.loc[x.index, 'Quantity_MT'] * df_processed.loc[x.index, 'Price_USD_per_kg']).sum() / df_processed.loc[x.index, 'Quantity_MT'].sum()
                }).reset_index().sort_values('Quarter')
                
                # Create dual-axis chart
                fig_quarterly = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_quarterly.add_trace(
                    go.Bar(
                        x=quarterly['Quarter'],
                        y=quarterly['Quantity_MT'],
                        name="Volume (MT)",
                        marker_color='#fb923c',
                        hovertemplate='<b>%{x}</b><br>Volume: %{y:.2f} MT<extra></extra>'
                    ),
                    secondary_y=False
                )
                
                fig_quarterly.add_trace(
                    go.Scatter(
                        x=quarterly['Quarter'],
                        y=quarterly['Price_USD_per_kg'],
                        name="Weighted Avg Price ($/kg)",
                        mode='lines+markers',
                        line=dict(color='#10b981', width=3),
                        marker=dict(size=10, color='#10b981'),
                        hovertemplate='<b>%{x}</b><br>Price: $%{y:.1f}/kg<extra></extra>'
                    ),
                    secondary_y=True
                )
                
                fig_quarterly.update_layout(
                    title="Quarterly Volume & Weighted Average Price",
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='rgba(255,255,255,0.05)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', size=14)
                )
                
                fig_quarterly.update_xaxes(title_text="Quarter", tickangle=45, title_font=dict(size=14))
                fig_quarterly.update_yaxes(title_text="<b>Volume (MT)</b>", secondary_y=False, gridcolor='rgba(251, 146, 60, 0.2)', title_font=dict(size=14, color='#fb923c'))
                fig_quarterly.update_yaxes(title_text="<b>Price ($/kg)</b>", secondary_y=True, gridcolor='rgba(16, 185, 129, 0.2)', title_font=dict(size=14, color='#10b981'))
                
                st.plotly_chart(fig_quarterly, use_container_width=True)
                
                # 2. Geographic Distribution
                st.markdown("---")
                st.markdown("### 🌍 Geographic Distribution")
                
                country_col = df_processed.attrs.get('country_col')
                if country_col and country_col in df_processed.columns:
                    geo_data = df_processed.groupby(country_col).agg({
                        'Quantity_MT': 'sum',
                        'Price_USD_per_kg': lambda x: (df_processed.loc[x.index, 'Quantity_MT'] * df_processed.loc[x.index, 'Price_USD_per_kg']).sum() / df_processed.loc[x.index, 'Quantity_MT'].sum()
                    }).reset_index()
                    
                    geo_data.columns = ['Country', 'Volume_MT', 'Avg_Price']
                    
                    fig_geo = px.choropleth(
                        geo_data,
                        locations='Country',
                        locationmode='country names',
                        color='Volume_MT',
                        hover_name='Country',
                        hover_data={'Volume_MT': ':.2f', 'Avg_Price': ':.1f'},
                        color_continuous_scale=[[0, '#fff7ed'], [0.5, '#fb923c'], [1, '#ea580c']],
                        title="Volume Distribution by Country (MT)"
                    )
                    
                    fig_geo.update_layout(
                         height=500,
                         geo=dict(
                         showframe=False,
                         showcoastlines=True,
                         projection_type='natural earth'
                         # Remove bgcolor from geo dict
                     ),
                         paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='#ffffff', size=14),
                         coloraxis_colorbar=dict(
                          title=dict(text="Volume (MT)", font=dict(size=14)),
                          tickfont=dict(size=12)
                         )
                      )
                    
                    st.plotly_chart(fig_geo, use_container_width=True)
                else:
                    st.info("No geographic data available in this dataset.")
                

                           
                                             
                



                # 3. Supply Chain Visualization
                st.markdown("---")
                st.markdown("### 🔗 Supply Chain Visualization")
                st.write("Interactive supply chain mapping showing flows between suppliers and buyers")
                
                # Financial year selector
                available_fys = sorted(df_processed['FY'].dropna().unique())
                
                if len(available_fys) > 0:
                    col_fy_select, col_fy_info = st.columns([1, 2])
                    
                    with col_fy_select:
                        selected_fy = st.selectbox(
                            "📅 Select Financial Year:",
                            options=available_fys,
                            key="fy_sankey_select"
                        )
                    
                    # Filter data for selected FY
                    fy_data = df_processed[df_processed['FY'] == selected_fy].copy()
                    
                    with col_fy_info:
                        st.write("")
                        st.write("")
                        st.info(f"📊 Showing data for **{selected_fy}** | {len(fy_data):,} transactions")
                    
                    # Determine data type and columns
                    data_type = dataset_info['type']
                    
                    # Find relevant columns based on data type
                    if data_type == 'Import':
                        source_label = 'Supplier'
                        target_label = 'Importer'
                        source_col_name = 'Supplier' if 'Supplier' in fy_data.columns else None
                        target_col_name = 'Importer' if 'Importer' in fy_data.columns else None
                        source_country_col = 'Country of Origin' if 'Country of Origin' in fy_data.columns else None
                        target_country_col = None
                        
                        # Fallback search if exact column names not found
                        if not source_col_name:
                            for col in fy_data.columns:
                                col_lower = col.lower()
                                if 'supplier' in col_lower and 'country' not in col_lower:
                                    source_col_name = col
                                    break
                        
                        if not target_col_name:
                            for col in fy_data.columns:
                                col_lower = col.lower()
                                if 'importer' in col_lower and 'country' not in col_lower:
                                    target_col_name = col
                                    break
                        
                    elif data_type == 'Export':
                        source_label = 'Exporter'
                        target_label = 'Foreign Buyer'
                        source_col_name = 'Exporter' if 'Exporter' in fy_data.columns else None
                        target_col_name = 'Foreign Buyer' if 'Foreign Buyer' in fy_data.columns else None
                        source_country_col = None
                        target_country_col = 'Country of Destination' if 'Country of Destination' in fy_data.columns else None
                        
                        # Fallback search if exact column names not found
                        if not source_col_name:
                            for col in fy_data.columns:
                                col_lower = col.lower()
                                if 'exporter' in col_lower and 'country' not in col_lower:
                                    source_col_name = col
                                    break
                        
                        if not target_col_name:
                            for col in fy_data.columns:
                                col_lower = col.lower()
                                if 'foreign' in col_lower and 'buyer' in col_lower and 'country' not in col_lower:
                                    target_col_name = col
                                    break
                        
                    else:  # Global
                        source_label = 'Supplier'
                        target_label = 'Buyer'
                        source_col_name = 'Supplier' if 'Supplier' in fy_data.columns else None
                        target_col_name = 'Buyer' if 'Buyer' in fy_data.columns else None
                        source_country_col = 'Supplier Country' if 'Supplier Country' in fy_data.columns else None
                        target_country_col = 'Buyer Country' if 'Buyer Country' in fy_data.columns else None
                        
                        # Fallback search if exact column names not found
                        if not source_col_name:
                            for col in fy_data.columns:
                                col_lower = col.lower()
                                if 'supplier' in col_lower and 'country' not in col_lower:
                                    source_col_name = col
                                    break
                        
                        if not target_col_name:
                            for col in fy_data.columns:
                                col_lower = col.lower()
                                if 'buyer' in col_lower and 'country' not in col_lower and 'foreign' not in col_lower:
                                    target_col_name = col
                                    break
                    
                    # Check if required columns exist
                    if source_col_name and target_col_name and source_col_name in fy_data.columns and target_col_name in fy_data.columns:
                        # Build Sankey data
                        nodes = []
                        node_dict = {}
                        links = []
                        
                        # Color palette
                        source_color = '#fb923c'  # Orange for suppliers/exporters
                        target_color = '#10b981'  # Green for buyers/importers
                        
                        # Get top entities for each side (limit to top 15 to avoid clutter)
                        top_sources = fy_data.groupby(source_col_name)['Quantity_MT'].sum().sort_values(ascending=False).head(15)
                        top_targets = fy_data.groupby(target_col_name)['Quantity_MT'].sum().sort_values(ascending=False).head(15)
                        
                        # Filter data to only top entities
                        filtered_data = fy_data[
                            (fy_data[source_col_name].isin(top_sources.index)) &
                            (fy_data[target_col_name].isin(top_targets.index))
                        ].copy()
                        
                        if len(filtered_data) > 0:
                            # Calculate aggregated data for each entity
                            agg_dict_source = {
                                'Quantity_MT': 'sum',
                                'Price_USD_per_kg': lambda x: (filtered_data.loc[x.index, 'Quantity_MT'] * x).sum() / filtered_data.loc[x.index, 'Quantity_MT'].sum()
                            }
                            if source_country_col and source_country_col in filtered_data.columns:
                                agg_dict_source[source_country_col] = 'first'
                            
                            source_data = filtered_data.groupby(source_col_name).agg(agg_dict_source).reset_index()
                            
                            agg_dict_target = {
                                'Quantity_MT': 'sum',
                                'Price_USD_per_kg': lambda x: (filtered_data.loc[x.index, 'Quantity_MT'] * x).sum() / filtered_data.loc[x.index, 'Quantity_MT'].sum()
                            }
                            if target_country_col and target_country_col in filtered_data.columns:
                                agg_dict_target[target_country_col] = 'first'
                            
                            target_data = filtered_data.groupby(target_col_name).agg(agg_dict_target).reset_index()
                            
                            # Build source nodes
                            for _, row in source_data.iterrows():
                                company = row[source_col_name]
                                volume = row['Quantity_MT']
                                price = row['Price_USD_per_kg']
                                country = row.get(source_country_col, 'N/A') if source_country_col and source_country_col in source_data.columns else 'N/A'
                                
                                node_key = f"SOURCE::{company}"
                                node_dict[node_key] = len(nodes)
                                
                                hover_text = f"<b>{company}</b><br>Country: {country}<br>Total Volume: {volume:.2f} MT<br>Avg Price: ${price:.2f}/kg"
                                nodes.append({
                                    'label': company,
                                    'color': source_color,
                                    'customdata': hover_text
                                })
                            
                            # Build target nodes
                            for _, row in target_data.iterrows():
                                company = row[target_col_name]
                                volume = row['Quantity_MT']
                                price = row['Price_USD_per_kg']
                                country = row.get(target_country_col, 'N/A') if target_country_col and target_country_col in target_data.columns else 'N/A'
                                
                                node_key = f"TARGET::{company}"
                                node_dict[node_key] = len(nodes)
                                
                                hover_text = f"<b>{company}</b><br>Country: {country}<br>Total Volume: {volume:.2f} MT<br>Avg Price: ${price:.2f}/kg"
                                nodes.append({
                                    'label': company,
                                    'color': target_color,
                                    'customdata': hover_text
                                })
                            
                            # Build links between sources and targets
                            flow_data = filtered_data.groupby([source_col_name, target_col_name])['Quantity_MT'].sum().reset_index()
                            
                            for _, row in flow_data.iterrows():
                                source_key = f"SOURCE::{row[source_col_name]}"
                                target_key = f"TARGET::{row[target_col_name]}"
                                
                                if source_key in node_dict and target_key in node_dict:
                                    links.append({
                                        'source': node_dict[source_key],
                                        'target': node_dict[target_key],
                                        'value': row['Quantity_MT'],
                                        'color': 'rgba(251, 146, 60, 0.25)'
                                    })
                            
                            # Create Sankey diagram
                            if len(links) > 0:
                                fig_sankey = go.Figure(data=[go.Sankey(
                                    arrangement='snap',
                                    node=dict(
                                        pad=20,
                                        thickness=20,
                                        line=dict(color='white', width=2),
                                        label=[node['label'] for node in nodes],
                                        color=[node['color'] for node in nodes],
                                        customdata=[node['customdata'] for node in nodes],
                                        hovertemplate='%{customdata}<extra></extra>'
                                    ),
                                    link=dict(
                                        source=[link['source'] for link in links],
                                        target=[link['target'] for link in links],
                                        value=[link['value'] for link in links],
                                        color=[link['color'] for link in links],
                                        hovertemplate='Flow: %{value:.2f} MT<extra></extra>'
                                    )
                                )])
                                
                                # Add column headers as annotations
                                fig_sankey.update_layout(
                                    title=f"Supply Chain Flow - {data_type} ({selected_fy})",
                                    font=dict(size=14, color='#ffffff', family='Poppins'),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    height=700,
                                    annotations=[
                                        dict(
                                            x=0.05,
                                            y=1.05,
                                            xref='paper',
                                            yref='paper',
                                            text=f'<b>{source_label}</b>',
                                            showarrow=False,
                                            font=dict(size=18, color='#fb923c', family='Poppins')
                                        ),
                                        dict(
                                            x=0.95,
                                            y=1.05,
                                            xref='paper',
                                            yref='paper',
                                            text=f'<b>{target_label}</b>',
                                            showarrow=False,
                                            font=dict(size=18, color='#10b981', family='Poppins')
                                        )
                                    ]
                                )
                                
                                st.plotly_chart(fig_sankey, use_container_width=True)
                                
                                # Insights Summary Panel
                                st.markdown("---")
                                st.markdown("#### 📊 Supply Chain Insights")
                                
                                col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)
                                
                                # Calculate insights
                                unique_sources = len(source_data)
                                unique_targets = len(target_data)
                                total_volume = filtered_data['Quantity_MT'].sum()
                                avg_price = (filtered_data['Quantity_MT'] * filtered_data['Price_USD_per_kg']).sum() / total_volume if total_volume > 0 else 0
                                
                                with col_insight1:
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, rgba(251, 146, 60, 0.2), rgba(249, 115, 22, 0.1)); 
                                                padding: 20px; border-radius: 12px; border: 2px solid #fb923c; text-align: center;'>
                                        <h3 style='color: #fb923c; margin: 0; font-size: 2rem;'>{unique_sources}</h3>
                                        <p style='color: #ffffff; margin: 5px 0 0 0; font-weight: 600;'>{source_label}s</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_insight2:
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1)); 
                                                padding: 20px; border-radius: 12px; border: 2px solid #10b981; text-align: center;'>
                                        <h3 style='color: #10b981; margin: 0; font-size: 2rem;'>{unique_targets}</h3>
                                        <p style='color: #ffffff; margin: 5px 0 0 0; font-weight: 600;'>{target_label}s</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_insight3:
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(124, 58, 237, 0.1)); 
                                                padding: 20px; border-radius: 12px; border: 2px solid #8b5cf6; text-align: center;'>
                                        <h3 style='color: #8b5cf6; margin: 0; font-size: 2rem;'>{total_volume:.0f}</h3>
                                        <p style='color: #ffffff; margin: 5px 0 0 0; font-weight: 600;'>Total Volume (MT)</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_insight4:
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, rgba(236, 72, 153, 0.2), rgba(219, 39, 119, 0.1)); 
                                                padding: 20px; border-radius: 12px; border: 2px solid #ec4899; text-align: center;'>
                                        <h3 style='color: #ec4899; margin: 0; font-size: 2rem;'>${avg_price:.2f}</h3>
                                        <p style='color: #ffffff; margin: 5px 0 0 0; font-weight: 600;'>Avg Price/kg</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Top Trading Pairs
                                st.markdown("---")
                                st.markdown("#### 🏆 Top 5 Trading Pairs")
                                
                                top_pairs = flow_data.sort_values('Quantity_MT', ascending=False).head(5)
                                
                                for idx, row in top_pairs.iterrows():
                                    col_pair, col_volume = st.columns([3, 1])
                                    
                                    with col_pair:
                                        st.markdown(f"""
                                        <div style='background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); 
                                                    padding: 15px; border-radius: 8px; border-left: 4px solid #fbbf24;'>
                                            <b style='color: #fb923c;'>{row[source_col_name]}</b> 
                                            <span style='color: #9ca3af;'>→</span> 
                                            <b style='color: #10b981;'>{row[target_col_name]}</b>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col_volume:
                                        st.markdown(f"""
                                        <div style='background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.1)); 
                                                    padding: 15px; border-radius: 8px; text-align: center;'>
                                            <b style='color: #fbbf24; font-size: 1.1rem;'>{row['Quantity_MT']:.2f} MT</b>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            else:
                                st.warning("⚠️ No valid flows found. Try selecting a different financial year.")
                        else:
                            st.warning("⚠️ Insufficient data after filtering to top entities.")
                    else:
                        st.warning("⚠️ Required columns for supply chain visualization not found in dataset.")
                        st.info(f"""
                        **Expected columns for {data_type} data:**
                        • {source_label}: {source_col_name if source_col_name else 'NOT FOUND'}
                        • {target_label}: {target_col_name if target_col_name else 'NOT FOUND'}
                        """)
                else:
                    st.info("No financial year data available. Ensure your dataset has valid date information.")
                
                # Download processed data
                st.markdown("---")
                st.markdown("### 📥 Export Processed Data")
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_processed.to_excel(writer, index=False, sheet_name='Processed Data')
                output.seek(0)
                
                st.download_button(
                    label="📥 Download Processed Analytics Data",
                    data=output,
                    file_name=f"analytics_{selected_dataset}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_analytics"
                )


        # Insert Customer & Competitor Intelligence tab content (analytics_tab3)
        with analytics_tab3:
                    st.markdown("""
                    ### 👥 Customer & Competitor Intelligence
                    Identify customers, competitors, and visualize supply chain networks for your target product
                    """)
        
                    # Initialize session state for customer analysis cache
                    if 'customer_analysis_cache' not in st.session_state:
                        st.session_state.customer_analysis_cache = {}
        
                    # Dataset Selection Section
                    st.markdown("---")
                    st.markdown("#### 📁 Select Datasets for Analysis")
        
                    col_sel1, col_sel2, col_sel3 = st.columns(3)
        
                    # Get available datasets by type
                    global_datasets = [name for name, info in st.session_state.saved_datasets.items()
                                       if info['type'] == 'Global']
                    import_datasets = [name for name, info in st.session_state.saved_datasets.items()
                                       if info['type'] == 'Import']
                    export_datasets = [name for name, info in st.session_state.saved_datasets.items()
                                       if info['type'] == 'Export']
        
                    with col_sel1:
                        st.markdown("**🌍 Global Trade Data**")
                        selected_rm = st.multiselect(
                            "Raw Materials",
                            options=global_datasets,
                            help="Select datasets containing raw material trade data",
                            key="rm_datasets"
                        )
        
                        selected_target = st.selectbox(
                            "Target Product",
                            options=["-- Select --"] + global_datasets,
                            help="Select ONE target product dataset",
                            key="target_dataset"
                        )
        
                        selected_downstreams = st.multiselect(
                            "Downstream Products",
                            options=global_datasets,
                            help="Select datasets containing downstream product data",
                            key="downstream_datasets"
                        )
        
                    with col_sel2:
                        st.markdown("**📥 Import Data (Optional)**")
                        selected_imports = st.multiselect(
                            "Import Datasets",
                            options=import_datasets,
                            help="Additional import data for customer identification",
                            key="import_datasets"
                        )
        
                    with col_sel3:
                        st.markdown("**📤 Export Data (Optional)**")
                        selected_exports = st.multiselect(
                            "Export Datasets",
                            options=export_datasets,
                            help="Export data for competitor identification",
                            key="export_datasets"
                        )
        
                    # Helper Functions
                    def extract_company_country_data(df, data_type='Global'):
                        """Extract company and country information from dataset"""
                        companies = []
        
                        # Find relevant columns (best-effort)
                        cols_lower = {c: c.lower() for c in df.columns}
        
                        if data_type == 'Global':
                            supplier_col = next((c for c in df.columns if 'supplier' in c.lower()), None)
                            buyer_col = next((c for c in df.columns if 'buyer' in c.lower() and 'foreign' not in c.lower()), None)
                            supplier_country_col = next((c for c in df.columns if 'supplier' in c.lower() and 'country' in c.lower()), None)
                            buyer_country_col = next((c for c in df.columns if 'buyer' in c.lower() and 'country' in c.lower()), None)
                        elif data_type == 'Import':
                            supplier_col = next((c for c in df.columns if 'supplier' in c.lower() or 'exporter' in c.lower()), None)
                            buyer_col = next((c for c in df.columns if 'importer' in c.lower() or 'buyer' in c.lower()), None)
                            supplier_country_col = next((c for c in df.columns if 'origin' in c.lower() or 'country' in c.lower()), None)
                            buyer_country_col = None
                        else:  # Export
                            supplier_col = next((c for c in df.columns if 'exporter' in c.lower() or 'supplier' in c.lower()), None)
                            buyer_col = next((c for c in df.columns if 'buyer' in c.lower() or 'foreign' in c.lower()), None)
                            supplier_country_col = None
                            buyer_country_col = next((c for c in df.columns if 'destination' in c.lower() or 'country' in c.lower()), None)
        
                        product_col = 'Standardized Name' if 'Standardized Name' in df.columns else None
                        qty_col = next((c for c in df.columns if 'quantity' in c.lower() or 'qty' in c.lower() or 'weight' in c.lower()), None)
                        value_col = next((c for c in df.columns if 'value' in c.lower() or 'amount' in c.lower() or 'fob' in c.lower() or 'price' in c.lower()), None)
        
                        if not product_col:
                            return []
        
                        # Iterate rows
                        for _, row in df.iterrows():
                            supplier = str(row[supplier_col]).strip() if supplier_col and pd.notna(row.get(supplier_col)) else None
                            buyer = str(row[buyer_col]).strip() if buyer_col and pd.notna(row.get(buyer_col)) else None
                            product = str(row[product_col]).strip() if pd.notna(row.get(product_col)) else None
        
                            supplier_country = str(row[supplier_country_col]).strip() if supplier_country_col and pd.notna(row.get(supplier_country_col)) else "Unknown"
                            buyer_country = str(row[buyer_country_col]).strip() if buyer_country_col and pd.notna(row.get(buyer_country_col)) else "Unknown"
        
                            qty = float(row[qty_col]) if qty_col and pd.notna(row.get(qty_col)) else 0.0
                            value = float(row[value_col]) if value_col and pd.notna(row.get(value_col)) else 0.0
        
                            if supplier and product:
                                companies.append({
                                    'company': supplier,
                                    'role': 'supplier',
                                    'product': product,
                                    'country': supplier_country,
                                    'quantity': qty,
                                    'value': value,
                                    'data_type': data_type
                                })
        
                            if buyer and product:
                                companies.append({
                                    'company': buyer,
                                    'role': 'buyer',
                                    'product': product,
                                    'country': buyer_country,
                                    'quantity': qty,
                                    'value': value,
                                    'data_type': data_type
                                })
        
                        return companies
        
                    def classify_companies(rm_data, target_data, downstream_data, import_data, export_data):
                        """Classify companies into customers, probable customers, and competitors"""
                        rm_companies = set(e['company'] for e in rm_data)
                        target_suppliers = set()
                        target_buyers = set()
                        downstream_suppliers = set()
                        downstream_buyers = set()
                        import_buyers = set(e['company'] for e in import_data if e['role'] == 'buyer')
                        export_suppliers = set(e['company'] for e in export_data if e['role'] == 'supplier')
        
                        target_product_name = None
                        for entry in target_data:
                            if target_product_name is None:
                                target_product_name = entry.get('product')
                            if entry['role'] == 'supplier':
                                target_suppliers.add(entry['company'])
                            else:
                                target_buyers.add(entry['company'])
        
                        for entry in downstream_data:
                            if entry['role'] == 'supplier':
                                downstream_suppliers.add(entry['company'])
                            else:
                                downstream_buyers.add(entry['company'])
        
                        # Sure-shot customers: direct buyers of target
                        sureshot_customers = target_buyers.copy()
        
                        # Probable customers: downstream actors not in target buyers or RM companies
                        probable_customers = {
                            c for c in (downstream_suppliers | downstream_buyers)
                            if c not in target_buyers and c not in rm_companies
                        }
        
                        # Competitors: suppliers of target + export suppliers
                        competitors = target_suppliers | export_suppliers
        
                        probable_competitors = rm_companies & target_buyers
        
                        all_data = rm_data + target_data + downstream_data + import_data + export_data
        
                        def create_company_details(company_set, category):
                            details = []
                            for company in sorted(company_set):
                                company_entries = [e for e in all_data if e['company'] == company]
                                if company_entries:
                                    total_qty = sum(e['quantity'] for e in company_entries)
                                    total_value = sum(e['value'] for e in company_entries)
                                    countries = list({e['country'] for e in company_entries if e.get('country') and e['country'] != 'Unknown'})
                                    products = list({e['product'] for e in company_entries if e.get('product')})
                                    details.append({
                                        'Company Name': company,
                                        'Country': countries[0] if countries else 'Unknown',
                                        'Category': category,
                                        'Products': ', '.join(list(products)[:3]),
                                        'Total Quantity (MT)': total_qty,
                                        'Total Value (USD)': total_value,
                                        'Source': 'Trade Data',
                                        'Verification Level': 'Direct' if category == 'Sure-shot Customer' else 'Probable'
                                    })
                            return details
        
                        sureshot_details = create_company_details(sureshot_customers, 'Sure-shot Customer')
                        probable_details = create_company_details(probable_customers, 'Probable Customer')
                        competitor_details = create_company_details(competitors | probable_competitors, 'Competitor')
        
                        return sureshot_details, probable_details, competitor_details, target_product_name
        
                    def create_supply_chain_network(rm_data, target_data, downstream_data):
                        """Create network graph for supply chain visualization"""
                        G = nx.DiGraph()
                        edge_data = []
        
                        # Build aggregated edges RM -> Target
                        rm_to_target = defaultdict(lambda: {'quantity': 0.0, 'value': 0.0, 'count': 0})
                        for rm in rm_data:
                            if rm['role'] == 'supplier':
                                for tgt in target_data:
                                    if tgt['role'] == 'buyer':
                                        key = (rm['company'], tgt['company'])
                                        rm_to_target[key]['quantity'] += rm['quantity']
                                        rm_to_target[key]['value'] += rm['value']
                                        rm_to_target[key]['count'] += 1
        
                        target_to_down = defaultdict(lambda: {'quantity': 0.0, 'value': 0.0, 'count': 0})
                        for tgt in target_data:
                            if tgt['role'] == 'supplier':
                                for ds in downstream_data:
                                    if ds['role'] == 'buyer':
                                        key = (tgt['company'], ds['company'])
                                        target_to_down[key]['quantity'] += tgt['quantity']
                                        target_to_down[key]['value'] += tgt['value']
                                        target_to_down[key]['count'] += 1
        
                        # Add edges to graph
                        for (s, t), data in rm_to_target.items():
                            if data['quantity'] > 0:
                                avg_price = (data['value'] / data['quantity']) if data['quantity'] > 0 else 0
                                G.add_edge(s, t, quantity=data['quantity'], value=data['value'], avg_price=avg_price, count=data['count'])
                                edge_data.append({
                                    'Source': s, 'Target': t, 'Type': 'RM → Target',
                                    'Quantity (MT)': data['quantity'], 'Value (USD)': data['value'], 'Avg Price (USD/kg)': avg_price, 'Transactions': data['count']
                                })
        
                        for (s, t), data in target_to_down.items():
                            if data['quantity'] > 0:
                                avg_price = (data['value'] / data['quantity']) if data['quantity'] > 0 else 0
                                G.add_edge(s, t, quantity=data['quantity'], value=data['value'], avg_price=avg_price, count=data['count'])
                                edge_data.append({
                                    'Source': s, 'Target': t, 'Type': 'Target → Downstream',
                                    'Quantity (MT)': data['quantity'], 'Value (USD)': data['value'], 'Avg Price (USD/kg)': avg_price, 'Transactions': data['count']
                                })
        
                        # Node info
                        node_info = {}
                        for node in set([n for e in G.edges() for n in e] + list(G.nodes())):
                            # Simple heuristic for node type
                            node_type = 'Unknown'
                            node_info[node] = {'type': node_type, 'color': '#3b82f6'}
        
                        return G, node_info, edge_data
        
                    # Analysis execution
                    st.markdown("---")
                    if st.button("🔍 Run Customer & Competitor Analysis", type="primary", use_container_width=True):
                        if selected_target == "-- Select --" or not selected_target:
                            st.error("⚠️ Please select a Target Product dataset")
                        elif not selected_rm and not selected_downstreams:
                            st.warning("⚠️ Please select at least Raw Materials or Downstream Products for comprehensive analysis")
                        else:
                            with st.spinner("🔄 Analyzing trade data and classifying companies..."):
                                try:
                                    rm_data = []
                                    for ds_name in selected_rm:
                                        df = st.session_state.saved_datasets[ds_name]['data']
                                        rm_data.extend(extract_company_country_data(df, 'Global'))
        
                                    target_data = []
                                    df_target = st.session_state.saved_datasets[selected_target]['data']
                                    target_data.extend(extract_company_country_data(df_target, 'Global'))
        
                                    downstream_data = []
                                    for ds_name in selected_downstreams:
                                        df = st.session_state.saved_datasets[ds_name]['data']
                                        downstream_data.extend(extract_company_country_data(df, 'Global'))
        
                                    import_data = []
                                    for ds_name in selected_imports:
                                        df = st.session_state.saved_datasets[ds_name]['data']
                                        import_data.extend(extract_company_country_data(df, 'Import'))
        
                                    export_data = []
                                    for ds_name in selected_exports:
                                        df = st.session_state.saved_datasets[ds_name]['data']
                                        export_data.extend(extract_company_country_data(df, 'Export'))
        
                                    sureshot, probable, competitors, target_product_name = classify_companies(
                                        rm_data, target_data, downstream_data, import_data, export_data
                                    )
        
                                    st.session_state.customer_analysis_cache = {
                                        'sureshot': sureshot,
                                        'probable': probable,
                                        'competitors': competitors,
                                        'target_product': target_product_name,
                                        'rm_data': rm_data,
                                        'target_data': target_data,
                                        'downstream_data': downstream_data
                                    }
        
                                    st.success(f"✅ Analysis complete for {target_product_name or selected_target}!")
                                    st.rerun()
        
                                except Exception as e:
                                    st.error(f"❌ Error during analysis: {str(e)}")
        
                    # Display cached results
                    if st.session_state.customer_analysis_cache:
                        cache = st.session_state.customer_analysis_cache
        
                        st.markdown("---")
                        st.markdown(f"### 📊 Analysis Results: {cache.get('target_product', 'Target Product')}")
        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
                        with col_m1:
                            st.markdown(f"**Sure-shot Customers**: {len(cache.get('sureshot', []))}")
                        with col_m2:
                            st.markdown(f"**Probable Customers**: {len(cache.get('probable', []))}")
                        with col_m3:
                            st.markdown(f"**Competitors**: {len(cache.get('competitors', []))}")
                        with col_m4:
                            total_companies = len(cache.get('sureshot', [])) + len(cache.get('probable', [])) + len(cache.get('competitors', []))
                            st.markdown(f"**Total Companies**: {total_companies}")
        
                        # Tabs for results
                        result_tab1, result_tab2, result_tab3, result_tab4, result_tab5 = st.tabs([
                            "✅ Sure-shot Customers",
                            "🎯 Probable Customers",
                            "⚔️ Competitors",
                            "🔗 Supply Chain Network",
                            "🌍 Geographic Heatmap"
                        ])
                        

                    # ...existing code...
                        with result_tab1:
                                        st.markdown("#### ✅ Sure-shot Customers")
                                        if cache.get('sureshot'):
                                            df_sureshot = pd.DataFrame(cache['sureshot'])
                                            # Rename Products column and add avg price per kg
                                            df_sureshot = df_sureshot.rename(columns={'Products': 'Products Imported'})
                                            df_sureshot['Avg Price (USD/kg)'] = df_sureshot.apply(
                                                lambda x: (x['Total Value (USD)'] / (x['Total Quantity (MT)'] * 1000)) 
                                                if x['Total Quantity (MT)'] > 0 else 0, axis=1
                                            )
                                            # Round to 2 decimal places
                                            df_sureshot['Avg Price (USD/kg)'] = df_sureshot['Avg Price (USD/kg)'].round(2)
                                            # Reorder columns to show price next to value
                                            cols = [col for col in df_sureshot.columns if col != 'Avg Price (USD/kg)']
                                            cols.insert(cols.index('Total Value (USD)') + 1, 'Avg Price (USD/kg)')
                                            df_sureshot = df_sureshot[cols]
                                            st.dataframe(df_sureshot, use_container_width=True, height=350)
                                        else:
                                            st.info("No sure-shot customers identified")
                    # ...existing code...                       


# ...existing code...
                        with result_tab2:
                            st.markdown("#### 🎯 Probable Customers")
                            if cache.get('probable'):
                                df_probable = pd.DataFrame(cache['probable'])
                                # Rename Products column and add avg price per kg
                                df_probable = df_probable.rename(columns={'Products': 'Products Manufactured'})
                                df_probable['Avg Price (USD/kg)'] = df_probable.apply(
                                    lambda x: (x['Total Value (USD)'] / (x['Total Quantity (MT)'] * 1000)) 
                                    if x['Total Quantity (MT)'] > 0 else 0, axis=1
                                )
                                # Round to 2 decimal places
                                df_probable['Avg Price (USD/kg)'] = df_probable['Avg Price (USD/kg)'].round(2)
                                # Reorder columns to show price next to value
                                cols = [col for col in df_probable.columns if col != 'Avg Price (USD/kg)']
                                cols.insert(cols.index('Total Value (USD)') + 1, 'Avg Price (USD/kg)')
                                df_probable = df_probable[cols]
                                st.dataframe(df_probable, use_container_width=True, height=350)
                            else:
                                st.info("No probable customers identified")
# ...existing code...


# ...existing code...
                        with result_tab3:
                            st.markdown("#### ⚔️ Competitors")
                            if cache.get('competitors'):
                                df_comp = pd.DataFrame(cache['competitors'])
                                # Rename Products column and add avg price per kg
                                df_comp = df_comp.rename(columns={'Products': 'Products Manufactured/Traded'})
                                df_comp['Avg Price (USD/kg)'] = df_comp.apply(
                                    lambda x: (x['Total Value (USD)'] / (x['Total Quantity (MT)'] * 1000)) 
                                    if x['Total Quantity (MT)'] > 0 else 0, axis=1
                                )
                                # Round to 2 decimal places
                                df_comp['Avg Price (USD/kg)'] = df_comp['Avg Price (USD/kg)'].round(2)
                                # Reorder columns to show price next to value
                                cols = [col for col in df_comp.columns if col != 'Avg Price (USD/kg)']
                                cols.insert(cols.index('Total Value (USD)') + 1, 'Avg Price (USD/kg)')
                                df_comp = df_comp[cols]
                                st.dataframe(df_comp, use_container_width=True, height=350)
                            else:
                                st.info("No competitors identified")
# ...existing code...



# ...existing code...
                        with result_tab4:
                            st.markdown("#### 🔗 Supply Chain Network")
                            try:
                                G, node_info, edge_data = create_supply_chain_network(
                                    cache.get('rm_data', []),
                                    cache.get('target_data', []),
                                    cache.get('downstream_data', [])
                                )
                                if edge_data:
                                    df_edges = pd.DataFrame(edge_data)
                                    
                                    # Add source and target locations by mapping from cache data
                                    all_data = (cache.get('rm_data', []) + 
                                              cache.get('target_data', []) + 
                                              cache.get('downstream_data', []))
                                    
                                    # Create company -> location mapping
                                    company_locations = {}
                                    for entry in all_data:
                                        if entry.get('company') and entry.get('country'):
                                            company_locations[entry['company']] = entry['country']
                                    
                                    # Add location columns
                                    df_edges['Source Location'] = df_edges['Source'].map(
                                        lambda x: company_locations.get(x, 'Unknown')
                                    )
                                    df_edges['Target Location'] = df_edges['Target'].map(
                                        lambda x: company_locations.get(x, 'Unknown')
                                    )
                                    
                                    # Reorder columns to show locations next to companies
                                    cols = ['Source', 'Source Location', 'Target', 'Target Location', 
                                           'Type', 'Quantity (MT)', 'Value (USD)', 'Avg Price (USD/kg)', 
                                           'Transactions']
                                    df_edges = df_edges[cols]
                                    
                                    st.dataframe(df_edges, use_container_width=True, height=350)
                                else:
                                    st.info("Insufficient data to create supply chain network")
                            except Exception as e:
                                st.error(f"Error creating supply chain visualization: {str(e)}")
# ...existing code...


                        with result_tab5:
                         st.markdown("#### 🌍 Geographic Distribution")
                         try:
                            # Create separate country data for each category
                            sureshot_country_data = defaultdict(lambda: {'quantity': 0.0, 'value': 0.0, 'companies': set()})
                            probable_country_data = defaultdict(lambda: {'quantity': 0.0, 'value': 0.0, 'companies': set()})
                            competitor_country_data = defaultdict(lambda: {'quantity': 0.0, 'value': 0.0, 'companies': set()})

                            # Process sureshot customers
                            for customer in cache.get('sureshot', []):
                                country = customer.get('Country')
                                if country and country != 'Unknown':
                                    sureshot_country_data[country]['quantity'] += customer['Total Quantity (MT)']
                                    sureshot_country_data[country]['value'] += customer['Total Value (USD)']
                                    sureshot_country_data[country]['companies'].add(customer['Company Name'])

                            # Process probable customers
                            for customer in cache.get('probable', []):
                                country = customer.get('Country')
                                if country and country != 'Unknown':
                                    probable_country_data[country]['quantity'] += customer['Total Quantity (MT)']
                                    probable_country_data[country]['value'] += customer['Total Value (USD)']
                                    probable_country_data[country]['companies'].add(customer['Company Name'])

                            # Process competitors
                            for competitor in cache.get('competitors', []):
                                country = competitor.get('Country')
                                if country and country != 'Unknown':
                                    competitor_country_data[country]['quantity'] += competitor['Total Quantity (MT)']
                                    competitor_country_data[country]['value'] += competitor['Total Value (USD)']
                                    competitor_country_data[country]['companies'].add(competitor['Company Name'])

                            # Create three columns for the heatmaps
                            col1, col2, col3 = st.columns(3)

                            # Sure-shot Customers Heatmap
                            with col1:
                                st.markdown("##### Sure-shot Customers")
                                if sureshot_country_data:
                                    df_sureshot = pd.DataFrame([
                                        {
                                            'Country': country,
                                            'Total Volume (MT)': data['quantity'],
                                            'Total Value (USD)': data['value'],
                                            'Number of Companies': len(data['companies'])
                                        }
                                        for country, data in sureshot_country_data.items()
                                    ]).sort_values('Total Volume (MT)', ascending=False)

                                    fig = px.choropleth(
                                        df_sureshot,
                                        locations='Country',
                                        locationmode='country names',
                                        color='Total Volume (MT)',
                                        hover_name='Country',
                                        color_continuous_scale='Oranges',
                                        title='Sure-shot Customer Distribution'
                                    )
                                    
                                    fig.update_layout(
                                          height=400, 
                                          margin=dict(l=0, r=0, t=30, b=0),
                                          geo=dict(
                                              showframe=False,
                                              showcoastlines=True,
                                              projection_type='natural earth'
                                         )
                                     )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.dataframe(df_sureshot, use_container_width=True, height=200)
                                else:
                                    st.info("No sure-shot customer geographic data available")

                            # Probable Customers Heatmap
                            with col2:
                                st.markdown("##### Probable Customers")
                                if probable_country_data:
                                    df_probable = pd.DataFrame([
                                        {
                                            'Country': country,
                                            'Total Volume (MT)': data['quantity'],
                                            'Total Value (USD)': data['value'],
                                            'Number of Companies': len(data['companies'])
                                        }
                                        for country, data in probable_country_data.items()
                                    ]).sort_values('Total Volume (MT)', ascending=False)

                                    fig = px.choropleth(
                                        df_probable,
                                        locations='Country',
                                        locationmode='country names',
                                        color='Total Volume (MT)',
                                        hover_name='Country',
                                        color_continuous_scale='Greens',
                                        title='Probable Customer Distribution'
                                    )
                                    fig.update_layout(
                                           height=400, 
                                           margin=dict(l=0, r=0, t=30, b=0),
                                               geo=dict(
                                                 showframe=False,
                                                 showcoastlines=True,
                                                 projection_type='natural earth'
                                             )
                                     )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.dataframe(df_probable, use_container_width=True, height=200)
                                else:
                                    st.info("No probable customer geographic data available")

                            # Competitors Heatmap
                            with col3:
                                st.markdown("##### Competitors")
                                if competitor_country_data:
                                    df_competitors = pd.DataFrame([
                                        {
                                            'Country': country,
                                            'Total Volume (MT)': data['quantity'],
                                            'Total Value (USD)': data['value'],
                                            'Number of Companies': len(data['companies'])
                                        }
                                        for country, data in competitor_country_data.items()
                                    ]).sort_values('Total Volume (MT)', ascending=False)

                                    fig = px.choropleth(
                                        df_competitors,
                                        locations='Country',
                                        locationmode='country names',
                                        color='Total Volume (MT)',
                                        hover_name='Country',
                                        color_continuous_scale='Reds',
                                        title='Competitor Distribution'
                                    )
                                    fig.update_layout(
                                            height=400, 
                                            margin=dict(l=0, r=0, t=30, b=0),
                                                geo=dict(
                                                    showframe=False,
                                                    showcoastlines=True,
                                                    projection_type='natural earth'
                                              )
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.dataframe(df_competitors, use_container_width=True, height=200)
                                else:
                                    st.info("No competitor geographic data available")

                         except Exception as e:
                            st.error(f"Error creating geographic visualizations: {str(e)}")                       



                # Add this inside analytics_tab4:
        
        with analytics_tab4:
            st.markdown("""
            ### 🔍 Company & Website Intelligence
            Search and classify chemical manufacturers based on product mentions and supply chain context
            """)
        
            # Input Section
            st.markdown("---")
            st.markdown("#### 🎯 Search Parameters")
        
            col1, col2 = st.columns(2)
            with col1:
                product_name = st.text_input(
                    "Product Name*", 
                    placeholder="e.g., Aniline",
                    help="Enter target product name"
                )
                
                cas_number = st.text_input(
                    "CAS Number (Optional)", 
                    placeholder="e.g., 62-53-3",
                    help="Enter CAS number for more precise matching"
                )
        
            with col2:
                raw_materials = st.text_area(
                    "Raw Materials (one per line)",
                    placeholder="e.g.,\nBenzene\nNitric Acid",
                    help="Enter raw materials, one per line"
                )
                
                downstreams = st.text_area(
                    "Downstream Products (one per line)",
                    placeholder="e.g.,\nParacetamol\nRubber Chemicals",
                    help="Enter downstream products, one per line"
                )
        
            # Process inputs
            rm_list = [x.strip() for x in raw_materials.split("\n") if x.strip()]
            ds_list = [x.strip() for x in downstreams.split("\n") if x.strip()]
        
            if st.button("🔎 Search & Classify Companies", type="primary", use_container_width=True):
                if not product_name:
                    st.error("⚠️ Please enter a product name")
                else:
                    with st.spinner("🔍 Searching and analyzing company websites... This may take 1-2 minutes."):
                        try:
                            # Perplexity API configuration
                            PERPLEXITY_API_KEY = "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"
                            
                            # Build search query
                            prompt = f"""You are an expert web-scraping and analysis assistant. Search and classify chemical manufacturers or distributors based on web mentions of:
        
        Product: {product_name}
        CAS: {cas_number if cas_number else 'Not provided'}
        Raw Materials: {', '.join(rm_list) if rm_list else 'None provided'}
        Downstream Products: {', '.join(ds_list) if ds_list else 'None provided'}
        
        Classification Logic:
        1. Direct Competitors: Explicitly mention product name/CAS
        2. Probable Customers: Mention downstream products OR raw materials + downstream
        3. Other/Low Confidence: Raw material mentions only
        
        Return 3 tables in Markdown format:
        1. Direct Competitors
        2. Probable Customers 
        3. Other/Low Confidence
        
        For each company include:
        - Company name
        - Website
        - Matched terms
        - Context snippet
        - Country
        - Confidence score (0-100)
        - Contact info (public only)
        
        Also provide:
        - Total counts summary
        - Top 5 most relevant probable customers
        
        Focus on manufacturer websites only. Exclude academic/wiki pages."""
        
                            # Make API request
                            headers = {
                                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                                "Content-Type": "application/json"
                            }
                            
                            payload = {
                                "model": "sonar",
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "You are a precise web intelligence assistant focused on chemical industry companies."
                                    },
                                    {
                                        "role": "user", 
                                        "content": prompt
                                    }
                                ],
                                "temperature": 0.1,
                                "max_tokens": 4000,
                                "return_citations": True
                            }
        
                            response = requests.post(
                                "https://api.perplexity.ai/chat/completions",
                                headers=headers,
                                json=payload,
                                timeout=120
                            )
        
                            if response.status_code == 200:
                                result = response.json()
                                
                                if 'choices' in result and len(result['choices']) > 0:
                                    company_data = result['choices'][0]['message']['content']
                                    
                                    # Display results
                                    st.success("✅ Analysis completed!")
                                    
                                    # Search parameters summary
                                    st.markdown("---")
                                    st.markdown("### 📋 Search Parameters")
                                    
                                    param_col1, param_col2 = st.columns(2)
                                    with param_col1:
                                        st.info(f"**Product:** {product_name}")
                                        if cas_number:
                                            st.info(f"**CAS:** {cas_number}")
                                    with param_col2:
                                        if rm_list:
                                            st.info(f"**Raw Materials:** {len(rm_list)}")
                                        if ds_list:
                                            st.info(f"**Downstreams:** {len(ds_list)}")
        
                                    # Display results
                                    st.markdown("---")
                                    st.markdown("### 📊 Company Classification Results")
                                    st.markdown(company_data)
        
                                    # Show citations
                                    if 'citations' in result and result['citations']:
                                        st.markdown("---")
                                        st.markdown("### 🔗 Sources")
                                        for idx, citation in enumerate(result['citations'], 1):
                                            st.markdown(f"{idx}. {citation}")
        
                                    # Download option
                                    st.markdown("---")
                                    st.download_button(
                                        label="📥 Download Results",
                                        data=company_data,
                                        file_name=f"company_analysis_{product_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                        mime="text/markdown"
                                    )
        
                                else:
                                    st.error("❌ No results returned from the API")
                            
                            else:
                                st.error(f"❌ API Error: {response.status_code}")
                                st.error(f"Response: {response.text}")
        
                        except requests.exceptions.Timeout:
                            st.error("⏱️ Request timed out. Please try again.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"🔴 Network Error: {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
            # Help section
            st.markdown("---")
            with st.expander("❓ How to Use Company Intelligence", expanded=False):
                st.markdown("""
                ### Guide to Company Intelligence Search
        
                **1. Input Parameters:**
                - **Product Name**: Your target chemical product (required)
                - **CAS Number**: For precise matching (optional)
                - **Raw Materials**: List of input chemicals
                - **Downstream Products**: End-use applications
        
                **2. Classification Logic:**
                - **Direct Competitors**: Explicitly mention your product
                - **Probable Customers**: Use your raw materials or make downstream products
                - **Other/Low Confidence**: Unclear or indirect mentions
        
                **3. Results Include:**
                - Company name and website
                - Matched keywords
                - Context snippets
                - Country location
                - Confidence score
                - Contact information
        
                **4. Tips:**
                - Include both chemical and trade names
                - Add important downstream applications
                - Check raw material connections
                - Review context snippets carefully
                """)



        with analytics_tab5:
            st.markdown("""
            ### 🔍 B2B Lead Generation & Enrichment
            Automatically enrich company data with contact information, websites, and decision-maker details
            """)
            
            # Check if customer analysis data exists
            if not st.session_state.customer_analysis_cache:
                st.info("📊 No customer data available. Please run Customer Intelligence analysis first (Tab: Customer Intelligence)")
                st.markdown("""
                ### How to use this feature:
                1. Go to **Customer & Competitor Intelligence** tab
                2. Select your datasets and run the analysis
                3. Come back here to enrich the customer data with contact information
                """)
            else:
                cache = st.session_state.customer_analysis_cache
                
                # Display available companies
                st.markdown("---")
                st.markdown("#### 📋 Available Companies for Enrichment")
                
                sureshot_companies = [c['Company Name'] for c in cache.get('sureshot', [])]
                probable_companies = [c['Company Name'] for c in cache.get('probable', [])]
                all_companies = sureshot_companies + probable_companies
                
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Sure-shot Customers", len(sureshot_companies))
                with col_info2:
                    st.metric("Probable Customers", len(probable_companies))
                with col_info3:
                    st.metric("Total Companies", len(all_companies))
                
                if not all_companies:
                    st.warning("No companies found. Please run Customer Intelligence analysis first.")
                else:
                    # Company selection
                    st.markdown("---")
                    st.markdown("#### 🎯 Select Companies to Enrich")
                    
                    col_select1, col_select2 = st.columns([3, 1])
                    
                    with col_select1:
                        select_all = st.checkbox("Select All Companies", value=True, key="select_all_companies")
                        
                        if select_all:
                            selected_companies = st.multiselect(
                                "Companies to enrich:",
                                options=all_companies,
                                default=all_companies[:10],  # Default to first 10 to avoid overload
                                help="Select companies to enrich with contact data"
                            )
                        else:
                            selected_companies = st.multiselect(
                                "Companies to enrich:",
                                options=all_companies,
                                help="Select companies to enrich with contact data"
                            )
                    
                    with col_select2:
                        st.write("")
                        st.write("")
                        batch_size = st.number_input(
                            "Batch Size",
                            min_value=1,
                            max_value=50,
                            value=5,
                            help="Process companies in batches to avoid rate limits"
                        )
                    
                    # Enrichment options
                    st.markdown("---")
                    st.markdown("#### ⚙️ Enrichment Options")
                    
                    col_opt1, col_opt2, col_opt3 = st.columns(3)
                    
                    with col_opt1:
                        search_emails = st.checkbox("Extract Emails from Websites", value=True)
                        search_phones = st.checkbox("Extract Phone Numbers", value=True)
                    
                    with col_opt2:
                        search_social = st.checkbox("Find Social Media Links", value=True)
                        search_location = st.checkbox("Extract Location Details", value=True)
                    
                    with col_opt3:
                        use_perplexity = st.checkbox(
                            "Use AI Search (Perplexity)", 
                            value=False,
                            help="Use Perplexity AI for enhanced contact discovery (requires API key)"
                        )
                    
                    # Start enrichment
                    st.markdown("---")
                    if st.button("🚀 Start Lead Enrichment", type="primary", use_container_width=True, disabled=not selected_companies):
                        
                        # Initialize progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results_container = st.empty()
                        
                        # Results storage
                        enriched_data = []
                        
                        try:
                            import re
                            import time
                            from urllib.parse import urlparse, quote_plus
                            
                            # Helper function to search Google for company info
                            def search_company_info(company_name, country=""):
                                """Search for company information using web search"""
                                try:
                                    search_query = f"{company_name} {country} official website contact"
                                    
                                    # Use requests to get basic info (no browser needed)
                                    headers = {
                                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                    }
                                    
                                    # Simple DuckDuckGo search (no API key needed)
                                    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
                                    
                                    response = requests.get(search_url, headers=headers, timeout=10)
                                    
                                    if response.status_code == 200:
                                        # Extract potential website URLs
                                        urls = re.findall(r'https?://(?:www\.)?([a-zA-Z0-9-]+\.(?:com|org|net|co|in|de|uk|fr|jp|cn))', response.text)
                                        
                                        # Filter for company website (exclude common domains)
                                        excluded = ['google', 'facebook', 'linkedin', 'twitter', 'youtube', 'wikipedia', 'amazon']
                                        company_urls = [url for url in urls if not any(exc in url.lower() for exc in excluded)]
                                        
                                        return company_urls[:3] if company_urls else []
                                    
                                    return []
                                
                                except Exception as e:
                                    return []
                            
                            # Helper function to extract emails from website
                            def extract_emails_from_website(url):
                                """Extract email addresses from a website"""
                                try:
                                    headers = {
                                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                    }
                                    
                                    response = requests.get(url, headers=headers, timeout=10)
                                    
                                    if response.status_code == 200:
                                        # Email regex pattern
                                        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                                        emails = re.findall(email_pattern, response.text)
                                        
                                        # Filter out common noise emails
                                        filtered_emails = [
                                            email for email in emails 
                                            if not any(x in email.lower() for x in ['example', 'test', 'placeholder', 'noreply', 'webmaster'])
                                        ]
                                        
                                        return list(set(filtered_emails))[:5]  # Return up to 5 unique emails
                                    
                                    return []
                                
                                except Exception as e:
                                    return []
                            
                            # Helper function to extract phones from website
                            def extract_phones_from_website(url):
                                """Extract phone numbers from a website"""
                                try:
                                    headers = {
                                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                    }
                                    
                                    response = requests.get(url, headers=headers, timeout=10)
                                    
                                    if response.status_code == 200:
                                        # Phone regex patterns (international)
                                        phone_patterns = [
                                            r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
                                            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
                                            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
                                        ]
                                        
                                        phones = []
                                        for pattern in phone_patterns:
                                            found = re.findall(pattern, response.text)
                                            phones.extend(found)
                                        
                                        # Clean and deduplicate
                                        cleaned_phones = list(set([p.strip() for p in phones if len(p) >= 10]))
                                        
                                        return cleaned_phones[:3]  # Return up to 3 phones
                                    
                                    return []
                                
                                except Exception as e:
                                    return []
                            
                            # Helper function to extract social media links
                            def extract_social_links(url):
                                """Extract social media profile links"""
                                try:
                                    headers = {
                                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                    }
                                    
                                    response = requests.get(url, headers=headers, timeout=10)
                                    
                                    if response.status_code == 200:
                                        social_patterns = {
                                            'LinkedIn': r'https?://(?:www\.)?linkedin\.com/(?:company|in)/[a-zA-Z0-9-]+',
                                            'Facebook': r'https?://(?:www\.)?facebook\.com/[a-zA-Z0-9.]+',
                                            'Twitter': r'https?://(?:www\.)?(?:twitter|x)\.com/[a-zA-Z0-9_]+',
                                            'Instagram': r'https?://(?:www\.)?instagram\.com/[a-zA-Z0-9._]+',
                                        }
                                        
                                        social_links = {}
                                        for platform, pattern in social_patterns.items():
                                            matches = re.findall(pattern, response.text)
                                            if matches:
                                                social_links[platform] = matches[0]
                                        
                                        return social_links
                                    
                                    return {}
                                
                                except Exception as e:
                                    return {}
                            
                            # Helper function using Perplexity AI
                            def search_with_perplexity(company_name, country=""):
                                """Use Perplexity AI to find comprehensive company info"""
                                if not use_perplexity:
                                    return None
                                
                                try:
                                    PERPLEXITY_API_KEY = "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"
                                    
                                    prompt = f"""Find contact information for the company "{company_name}" {f"located in {country}" if country else ""}.

Provide:
1. Official website URL
2. Business email addresses
3. Phone numbers
4. Physical address
5. LinkedIn company page
6. Key decision maker contacts (if available)

Format response as JSON with keys: website, emails, phones, address, linkedin, contacts"""

                                    headers = {
                                        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                                        "Content-Type": "application/json"
                                    }
                                    
                                    payload = {
                                        "model": "sonar",
                                        "messages": [
                                            {
                                                "role": "user",
                                                "content": prompt
                                            }
                                        ],
                                        "temperature": 0.1,
                                        "max_tokens": 1000
                                    }
                                    
                                    response = requests.post(
                                        "https://api.perplexity.ai/chat/completions",
                                        headers=headers,
                                        json=payload,
                                        timeout=30
                                    )
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        if 'choices' in result and len(result['choices']) > 0:
                                            content = result['choices'][0]['message']['content']
                                            
                                            # Try to extract structured data
                                            try:
                                                import json
                                                # Look for JSON in the response
                                                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                                                if json_match:
                                                    data = json.loads(json_match.group())
                                                    return data
                                            except:
                                                pass
                                            
                                            # Fallback: extract from text
                                            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
                                            phones = re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', content)
                                            urls = re.findall(r'https?://[^\s<>"]+', content)
                                            
                                            return {
                                                'website': urls[0] if urls else None,
                                                'emails': emails[:3],
                                                'phones': phones[:2],
                                                'raw_text': content
                                            }
                                    
                                    return None
                                
                                except Exception as e:
                                    return None
                            
                            # Process companies in batches
                            total_companies = len(selected_companies)
                            
                            for idx, company_name in enumerate(selected_companies):
                                # Update progress
                                progress = (idx + 1) / total_companies
                                progress_bar.progress(progress)
                                status_text.text(f"🔍 Processing {idx + 1}/{total_companies}: {company_name}")
                                
                                # Get company details from cache
                                company_details = None
                                for c in cache.get('sureshot', []) + cache.get('probable', []):
                                    if c['Company Name'] == company_name:
                                        company_details = c
                                        break
                                
                                country = company_details.get('Country', '') if company_details else ''
                                category = company_details.get('Category', 'Unknown') if company_details else 'Unknown'
                                
                                # Initialize result
                                result = {
                                    'Company Name': company_name,
                                    'Country': country,
                                    'Category': category,
                                    'Website': None,
                                    'Emails': [],
                                    'Phones': [],
                                    'Address': None,
                                    'LinkedIn': None,
                                    'Social Media': {},
                                    'Status': 'Processing'
                                }
                                
                                try:
                                    # Method 1: Try Perplexity AI if enabled
                                    if use_perplexity:
                                        status_text.text(f"🤖 Using AI search for {company_name}...")
                                        perplexity_data = search_with_perplexity(company_name, country)
                                        
                                        if perplexity_data:
                                            result['Website'] = perplexity_data.get('website')
                                            result['Emails'] = perplexity_data.get('emails', [])
                                            result['Phones'] = perplexity_data.get('phones', [])
                                            result['Address'] = perplexity_data.get('address')
                                            result['LinkedIn'] = perplexity_data.get('linkedin')
                                            result['Status'] = 'Enriched (AI)'
                                    
                                    # Method 2: Web search for website
                                    if not result['Website']:
                                        status_text.text(f"🌐 Searching web for {company_name}...")
                                        potential_urls = search_company_info(company_name, country)
                                        
                                        if potential_urls:
                                            result['Website'] = f"https://www.{potential_urls[0]}"
                                    
                                    # Method 3: Extract from website if found
                                    if result['Website']:
                                        status_text.text(f"📧 Extracting contacts from website...")
                                        
                                        # Extract emails
                                        if search_emails and not result['Emails']:
                                            emails = extract_emails_from_website(result['Website'])
                                            result['Emails'] = emails
                                        
                                        # Extract phones
                                        if search_phones and not result['Phones']:
                                            phones = extract_phones_from_website(result['Website'])
                                            result['Phones'] = phones
                                        
                                        # Extract social media
                                        if search_social:
                                            social = extract_social_links(result['Website'])
                                            result['Social Media'] = social
                                            if 'LinkedIn' in social:
                                                result['LinkedIn'] = social['LinkedIn']
                                        
                                        result['Status'] = 'Enriched'
                                    else:
                                        result['Status'] = 'No Website Found'
                                
                                except Exception as e:
                                    result['Status'] = f'Error: {str(e)}'
                                
                                enriched_data.append(result)
                                
                                # Rate limiting
                                if (idx + 1) % batch_size == 0:
                                    status_text.text(f"⏸️ Batch complete, pausing for 3 seconds...")
                                    time.sleep(3)
                                else:
                                    time.sleep(1)
                                
                                # Show intermediate results
                                with results_container.container():
                                    st.markdown(f"**Progress:** {idx + 1}/{total_companies} companies processed")
                                    
                                    # Quick stats
                                    enriched_count = sum(1 for r in enriched_data if r['Status'] == 'Enriched' or r['Status'] == 'Enriched (AI)')
                                    with_emails = sum(1 for r in enriched_data if r['Emails'])
                                    with_phones = sum(1 for r in enriched_data if r['Phones'])
                                    
                                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                    col_stat1.metric("Processed", f"{idx + 1}")
                                    col_stat2.metric("Enriched", enriched_count)
                                    col_stat3.metric("With Emails", with_emails)
                                    col_stat4.metric("With Phones", with_phones)
                            
                            # Final results
                            progress_bar.progress(1.0)
                            status_text.text("✅ Enrichment complete!")
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("### 📊 Enrichment Results")
                            
                            # Summary metrics
                            total_enriched = sum(1 for r in enriched_data if r['Status'] in ['Enriched', 'Enriched (AI)'])
                            total_emails = sum(len(r['Emails']) for r in enriched_data)
                            total_phones = sum(len(r['Phones']) for r in enriched_data)
                            total_websites = sum(1 for r in enriched_data if r['Website'])
                            
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            
                            with col_m1:
                                st.metric("Successfully Enriched", f"{total_enriched}/{total_companies}")
                            with col_m2:
                                st.metric("Total Emails Found", total_emails)
                            with col_m3:
                                st.metric("Total Phones Found", total_phones)
                            with col_m4:
                                st.metric("Websites Found", total_websites)
                            
                            # Create detailed results table
                            results_df = pd.DataFrame([
                                {
                                    'Company Name': r['Company Name'],
                                    'Country': r['Country'],
                                    'Category': r['Category'],
                                    'Website': r['Website'] or 'Not Found',
                                    'Emails': ', '.join(r['Emails']) if r['Emails'] else 'None',
                                    'Phones': ', '.join(r['Phones']) if r['Phones'] else 'None',
                                    'LinkedIn': r['LinkedIn'] or 'Not Found',
                                    'Status': r['Status']
                                }
                                for r in enriched_data
                            ])
                            
                            st.dataframe(results_df, use_container_width=True, height=400)
                            
                            # Download options
                            st.markdown("---")
                            st.markdown("### 📥 Export Results")
                            
                            col_dl1, col_dl2 = st.columns(2)
                            
                            with col_dl1:
                                # Excel export
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    results_df.to_excel(writer, sheet_name='Enriched Leads', index=False)
                                    
                                    # Add detailed sheet with social media
                                    detailed_df = pd.DataFrame([
                                        {
                                            'Company': r['Company Name'],
                                            'Country': r['Country'],
                                            'Category': r['Category'],
                                            'Website': r['Website'] or '',
                                            'Email 1': r['Emails'][0] if len(r['Emails']) > 0 else '',
                                            'Email 2': r['Emails'][1] if len(r['Emails']) > 1 else '',
                                            'Email 3': r['Emails'][2] if len(r['Emails']) > 2 else '',
                                            'Phone 1': r['Phones'][0] if len(r['Phones']) > 0 else '',
                                            'Phone 2': r['Phones'][1] if len(r['Phones']) > 1 else '',
                                            'LinkedIn': r['LinkedIn'] or '',
                                            'Facebook': r['Social Media'].get('Facebook', ''),
                                            'Twitter': r['Social Media'].get('Twitter', ''),
                                            'Status': r['Status']
                                        }
                                        for r in enriched_data
                                    ])
                                    detailed_df.to_excel(writer, sheet_name='Detailed Contacts', index=False)
                                
                                output.seek(0)
                                
                                st.download_button(
                                    label="📥 Download Excel Report",
                                    data=output,
                                    file_name=f"enriched_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            with col_dl2:
                                # CSV export
                                csv_output = results_df.to_csv(index=False)
                                
                                st.download_button(
                                    label="📄 Download CSV",
                                    data=csv_output,
                                    file_name=f"enriched_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            # Success tips
                            st.markdown("---")
                            st.success("""
                            ✅ **Next Steps:**
                            1. Review the enriched data in the Excel file
                            2. Verify email addresses and phone numbers
                            3. Use LinkedIn links for direct outreach
                            4. Import to your CRM system
                            """)
                        
                        except Exception as e:
                            st.error(f"❌ Error during enrichment: {str(e)}")
                            with st.expander("View Error Details"):
                                st.exception(e)
                        
                        finally:
                            progress_bar.empty()
                            status_text.empty()
            
            # Help section
            st.markdown("---")
            with st.expander("❓ How to Use B2B Lead Enrichment", expanded=False):
                st.markdown("""
                ### 🎯 Lead Enrichment Guide
                
                **Step-by-Step Process:**
                
                1. **Prerequisites:**
                   - Run Customer Intelligence analysis first
                   - Ensure you have sure-shot and probable customers identified
                
                2. **Select Companies:**
                   - Choose which companies to enrich
                   - Use batch processing to avoid rate limits (recommended: 5-10 companies per batch)
                
                3. **Choose Enrichment Options:**
                   - **Extract Emails**: Find email addresses from company websites
                   - **Extract Phones**: Discover phone numbers from contact pages
                   - **Social Media**: Find LinkedIn, Facebook, Twitter profiles
                   - **AI Search**: Use Perplexity AI for enhanced discovery (optional)
                
                4. **Review Results:**
                   - Check enrichment status for each company
                   - Verify contact information accuracy
                   - Note companies that need manual research
                
                5. **Export Data:**
                   - Download Excel with detailed contacts
                   - Import to CRM or email marketing tool
                   - Use for targeted outreach campaigns
                
                **Data Sources:**
                - Web search (DuckDuckGo)
                - Company websites (direct scraping)
                - Public contact pages
                - Social media profiles
                - AI-powered search (Perplexity - optional)
                
                **Privacy & Compliance:**
                - Only uses publicly available information
                - No login or authentication required
                - Respects robots.txt and rate limits
                - All data is from public sources
                
                **Tips for Best Results:**
                - Start with small batches (5-10 companies)
                - Enable AI search for better accuracy
                - Manually verify critical contacts
                - Use during off-peak hours to avoid rate limits
                - Cross-reference multiple data sources
                
                **Common Issues:**
                - **No Website Found**: Company may not have online presence
                - **No Contacts Found**: Try enabling AI search or check manually
                - **Rate Limit**: Increase delay between requests
                - **Timeout**: Some websites may be slow or block scrapers
                """)
            

            

            






with tab5:
    st.markdown("""
    ### 📊 Market Estimation For Last FY or Last Surveyed year
    Estimate total market demand for your target product based on downstream applications.
    """)        


    
    # Initialize session state for downstreams
    if 'downstreams' not in st.session_state:
        st.session_state.downstreams = []
    if 'downstream_counter' not in st.session_state:
        st.session_state.downstream_counter = 0
    
    # Helper function to parse numeric input (handle NA, blank, or valid numbers)
    def parse_numeric_input(value):
        """Convert input to float, treating NA/blank as 0"""
        if value is None or value == "" or str(value).strip().upper() == "NA":
            return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0
    
    # Section 1: Target Product Input
    st.markdown("---")
    st.markdown("#### 🎯 Target Product")
    
    target_product = st.text_input(
        "Enter Target Product Name",
        placeholder="e.g., Aniline, Benzene, etc.",
        key="target_product_input",
        help="The main product for which you want to estimate market demand"
    )
    
    if target_product:
        st.success(f"✅ Target Product: **{target_product}**")
        
        # Section 2: Downstream Applications
        st.markdown("---")
        st.markdown("#### 📦 Downstream Applications")
        st.write("Add downstream applications that consume your target product")
        
        # Add new downstream form
        with st.expander("➕ Add New Downstream Application", expanded=len(st.session_state.downstreams) == 0):
            col_name, col_category = st.columns([2, 1])
            
            with col_name:
                downstream_name = st.text_input(
                    "Downstream Product/Application Name",
                    placeholder="e.g., Paracetamol, Glyphosate, Rubber Antioxidant",
                    key="new_downstream_name"
                )
            
            with col_category:
                downstream_category = st.selectbox(
                    "Category",
                    options=["Pharma", "Agro", "Others"],
                    key="new_downstream_category"
                )
            
            # Manual entry inputs for all categories
            col_demand, col_norm = st.columns(2)
            
            with col_demand:
                demand_input = st.text_input(
                    "Demand (MT)",
                    placeholder="Enter demand or 'NA'",
                    key="demand_input",
                    help="Enter the demand for this downstream product in metric tons"
                )
            
            with col_norm:
                norm_input = st.text_input(
                    "Norm (MT per MT of Target Product)",
                    placeholder="e.g., 0.85 or 'NA'",
                    key="norm_input",
                    help="How many MT of target product needed per MT of downstream product"
                )
            
            # Add downstream button
            if st.button("➕ Add Downstream", key="add_downstream_btn"):
                if downstream_name:
                    # Parse inputs
                    demand = parse_numeric_input(demand_input)
                    norm = parse_numeric_input(norm_input)
                    
                    # Create downstream entry
                    new_downstream = {
                        'id': st.session_state.downstream_counter,
                        'name': downstream_name,
                        'category': downstream_category,
                        'demand_mt': demand,
                        'norm': norm,
                        'calculated_demand': demand * norm
                    }
                    
                    st.session_state.downstreams.append(new_downstream)
                    st.session_state.downstream_counter += 1
                    
                    st.success(f"✅ Added: {downstream_name} ({downstream_category})")
                    st.rerun()
                else:
                    st.error("⚠️ Please enter a downstream product name")
        
        # Display existing downstreams
        if st.session_state.downstreams:
            st.markdown("---")
            st.markdown("#### 📋 Added Downstreams")
            
            # Create display table
            display_data = []
            for ds in st.session_state.downstreams:
                display_data.append({
                    'Downstream Product': ds['name'],
                    'Category': ds['category'],
                    'Demand (MT)': f"{ds['demand_mt']:,.2f}" if ds['demand_mt'] > 0 else "NA",
                    'Norm (MT/MT)': f"{ds['norm']:.4f}" if ds['norm'] > 0 else "NA",
                    'Calculated Demand (MT)': f"{ds['calculated_demand']:,.2f}" if ds['calculated_demand'] > 0 else "NA"
                })
            
            df_display = pd.DataFrame(display_data)
            st.dataframe(df_display, use_container_width=True)
            
            # Management options
            col_manage1, col_manage2 = st.columns([3, 1])
            
            with col_manage1:
                selected_to_remove = st.selectbox(
                    "Select downstream to remove:",
                    options=["-- Select --"] + [ds['name'] for ds in st.session_state.downstreams],
                    key="remove_downstream_select"
                )
            
            with col_manage2:
                st.write("")
                st.write("")
                if st.button("🗑️ Remove", key="remove_downstream_btn"):
                    if selected_to_remove != "-- Select --":
                        st.session_state.downstreams = [
                            ds for ds in st.session_state.downstreams 
                            if ds['name'] != selected_to_remove
                        ]
                        st.success(f"Removed: {selected_to_remove}")
                        st.rerun()
            
            # Calculate total market estimation
            st.markdown("---")
            st.markdown("#### 🎯 Market Estimation Calculation")
            
            if st.button("📊 Calculate Market Estimation", type="primary", use_container_width=True):
                st.markdown("---")
                st.markdown("### 📈 Results")
                
                # Category-wise breakdown
                pharma_demand = sum([ds['calculated_demand'] for ds in st.session_state.downstreams if ds['category'] == 'Pharma'])
                agro_demand = sum([ds['calculated_demand'] for ds in st.session_state.downstreams if ds['category'] == 'Agro'])
                others_demand = sum([ds['calculated_demand'] for ds in st.session_state.downstreams if ds['category'] == 'Others'])
                total_demand = pharma_demand + agro_demand + others_demand
                
                # Display breakdown by category
                st.markdown("##### 📊 Demand Breakdown by Category")
                
                col_cat1, col_cat2, col_cat3, col_cat4 = st.columns(4)
                
                with col_cat1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.1)); 
                                padding: 20px; border-radius: 12px; border: 2px solid #fbbf24; text-align: center;'>
                        <h3 style='color: #fbbf24; margin: 0; font-size: 1.8rem;'>{pharma_demand:,.2f}</h3>
                        <p style='color: #ffffff; margin: 5px 0 0 0; font-weight: 600;'>Pharma (MT)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_cat2:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.1)); 
                                padding: 20px; border-radius: 12px; border: 2px solid #22c55e; text-align: center;'>
                        <h3 style='color: #22c55e; margin: 0; font-size: 1.8rem;'>{agro_demand:,.2f}</h3>
                        <p style='color: #ffffff; margin: 5px 0 0 0; font-weight: 600;'>Agro (MT)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_cat3:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(124, 58, 237, 0.1)); 
                                padding: 20px; border-radius: 12px; border: 2px solid #8b5cf6; text-align: center;'>
                        <h3 style='color: #8b5cf6; margin: 0; font-size: 1.8rem;'>{others_demand:,.2f}</h3>
                        <p style='color: #ffffff; margin: 5px 0 0 0; font-weight: 600;'>Others (MT)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_cat4:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(220, 38, 38, 0.15)); 
                                padding: 20px; border-radius: 12px; border: 3px solid #ef4444; text-align: center; 
                                box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4);'>
                        <h3 style='color: #ef4444; margin: 0; font-size: 2rem; font-weight: 800;'>{total_demand:,.2f}</h3>
                        <p style='color: #ffffff; margin: 5px 0 0 0; font-weight: 700; text-transform: uppercase;'>TOTAL DEMAND (MT)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed breakdown per downstream
                st.markdown("---")
                st.markdown("##### 📋 Detailed Breakdown by Downstream")
                
                breakdown_data = []
                for ds in st.session_state.downstreams:
                    breakdown_data.append({
                        'Downstream Product': ds['name'],
                        'Category': ds['category'],
                        'Demand (MT)': ds['demand_mt'],
                        'Norm (MT/MT)': ds['norm'],
                        'Calculated Demand (MT)': ds['calculated_demand'],
                        'Contribution %': (ds['calculated_demand'] / total_demand * 100) if total_demand > 0 else 0
                    })
                
                df_breakdown = pd.DataFrame(breakdown_data)
                
                # Format the dataframe
                df_breakdown['Demand (MT)'] = df_breakdown['Demand (MT)'].apply(lambda x: f"{x:,.2f}" if x > 0 else "NA")
                df_breakdown['Norm (MT/MT)'] = df_breakdown['Norm (MT/MT)'].apply(lambda x: f"{x:.4f}" if x > 0 else "NA")
                df_breakdown['Calculated Demand (MT)'] = df_breakdown['Calculated Demand (MT)'].apply(lambda x: f"{x:,.2f}")
                df_breakdown['Contribution %'] = df_breakdown['Contribution %'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(df_breakdown, use_container_width=True)
                
                # Visual representation - Pie chart
                st.markdown("---")
                st.markdown("##### 🥧 Demand Distribution")
                
                if total_demand > 0:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=[ds['name'] for ds in st.session_state.downstreams],
                        values=[ds['calculated_demand'] for ds in st.session_state.downstreams],
                        hole=0.4,
                        marker=dict(
                            colors=['#fbbf24', '#22c55e', '#8b5cf6', '#ef4444', '#3b82f6', '#ec4899', '#14b8a6', '#f97316'],
                            line=dict(color='#ffffff', width=2)
                        ),
                        textposition='auto',
                        textinfo='label+percent',
                        hovertemplate='<b>%{label}</b><br>Demand: %{value:,.2f} MT<br>Share: %{percent}<extra></extra>'
                    )])
                    
                    fig_pie.update_layout(
                        title=f"Market Distribution for {target_product}",
                        font=dict(size=14, color='#ffffff', family='Poppins'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=500,
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="middle",
                            y=0.5,
                            xanchor="left",
                            x=1.05,
                            bgcolor='rgba(255,255,255,0.05)',
                            bordercolor='rgba(251, 146, 60, 0.3)',
                            borderwidth=2
                        )
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Summary report
                st.markdown("---")
                st.markdown("##### 📄 Summary Report")
                
                summary_text = f"""
**Market Estimation Summary for {target_product}**

**Total Estimated Demand:** {total_demand:,.2f} MT

**Breakdown by Category:**
- Pharmaceuticals: {pharma_demand:,.2f} MT ({(pharma_demand/total_demand*100) if total_demand > 0 else 0:.1f}%)
- Agrochemicals: {agro_demand:,.2f} MT ({(agro_demand/total_demand*100) if total_demand > 0 else 0:.1f}%)
- Others: {others_demand:,.2f} MT ({(others_demand/total_demand*100) if total_demand > 0 else 0:.1f}%)

**Number of Downstream Applications:** {len(st.session_state.downstreams)}

**Top 3 Contributors:**
"""
                # Sort by calculated demand
                sorted_downstreams = sorted(st.session_state.downstreams, key=lambda x: x['calculated_demand'], reverse=True)
                for i, ds in enumerate(sorted_downstreams[:3], 1):
                    contribution_pct = (ds['calculated_demand'] / total_demand * 100) if total_demand > 0 else 0
                    summary_text += f"\n{i}. {ds['name']} ({ds['category']}): {ds['calculated_demand']:,.2f} MT ({contribution_pct:.1f}%)"
                
                st.markdown(summary_text)
                
                # Download options
                st.markdown("---")
                st.markdown("##### 📥 Export Results")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    # Excel export
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Summary sheet
                        summary_df = pd.DataFrame({
                            'Target Product': [target_product],
                            'Total Demand (MT)': [total_demand],
                            'Pharma Demand (MT)': [pharma_demand],
                            'Agro Demand (MT)': [agro_demand],
                            'Others Demand (MT)': [others_demand],
                            'Number of Downstreams': [len(st.session_state.downstreams)]
                        })
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Detailed breakdown sheet
                        detail_df = pd.DataFrame(breakdown_data)
                        detail_df.to_excel(writer, sheet_name='Detailed Breakdown', index=False)
                    
                    output.seek(0)
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=output,
                        file_name=f"market_estimation_{target_product.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col_dl2:
                    # Text report export
                    st.download_button(
                        label="📄 Download Text Summary",
                        data=summary_text,
                        file_name=f"market_summary_{target_product.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
        
        else:
            st.info("👆 Add downstream applications to begin market estimation")
    
    else:
        st.info("👆 Enter a target product name to begin")
    
    # Help section
    st.markdown("---")
    with st.expander("ℹ️ How to Use Market Estimation", expanded=False):
        st.markdown("""
        **Step-by-Step Guide:**
        
        1. **Enter Target Product:** Input the name of the product for which you want to estimate market demand
        
        2. **Add Downstream Applications:** For each downstream application that uses your target product:
           - Enter the downstream product/application name
           - Select category (Pharma/Agro/Others)
           - Enter demand in metric tons (MT)
           - Enter norm (consumption ratio): How many MT of target product are needed per MT of downstream product
        
        3. **Review Added Downstreams:** Check the table of added applications and remove any if needed
        
        4. **Calculate:** Click the "Calculate Market Estimation" button to see:
           - Total estimated demand for your target product
           - Category-wise breakdown (Pharma/Agro/Others)
           - Detailed contribution from each downstream
           - Visual distribution charts
           - Downloadable reports
        
        **Tips:**
        - Enter "NA" or leave blank for any missing information (treated as 0)
        - Norm values are typically between 0 and 2 (e.g., 0.85 means 0.85 MT of target needed per 1 MT of downstream)
        - You can add multiple downstreams in the same or different categories
        - The calculated demand = Downstream Demand × Norm
        
        **Example:**
        - Target Product: Aniline
        - Downstream: Paracetamol (Pharma)
          - Demand: 12,500 MT
          - Norm: 0.85 MT/MT
          - Calculated: 10,625 MT of Aniline needed
        """)

with tab6:
             st.subheader("🔬 Environmental Clearance Analysis")
             st.write("Search for environmental clearances related to chemical products using AI-powered research")
    
            
        
             # Input section
             col_input1, col_input2 = st.columns(2)
        
             with col_input1:
              product_name = st.text_input(
                "🧪 Product Name",
                placeholder="e.g., Tiafenacil, Aniline, etc.",
                help="Enter the chemical product name you want to search for"
             )
        
             with col_input2:
               cas_number = st.text_input(
                "🔢 CAS Number (Optional)",
                placeholder="e.g., 1228284-64-7",
                help="Enter CAS number for more precise results"
             )
        
             # Search button
             search_button = st.button("🔎 Search Environmental Clearances", type="primary", use_container_width=True)
        
             if search_button:
              if not product_name and not cas_number:
                st.error("⚠️ Please enter either a product name or CAS number to search.")
             else:
                with st.spinner("🔍 Searching environmental clearance databases... This may take 30-60 seconds."):
                    try:
                        # Perplexity API configuration
                        PERPLEXITY_API_KEY = "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"
                        
                        # Build search query
                        search_terms = []
                        if product_name:
                            search_terms.append(f'product name: "{product_name}"')
                        if cas_number:
                            search_terms.append(f'CAS number: "{cas_number}"')
                        
                        search_query = " OR ".join(search_terms)
                        
                        # Create the prompt for Perplexity
                        prompt = f"""You are an expert data extraction assistant specializing in environmental compliance research. Task: Find all publicly available environmental clearance (EC) documents on the internet that mention the given chemical product or CAS number only don't include names related to it, find exact names of this only don't include products name related to it. Apart from searching the whole internet for this also focus on government portals such as PARIVESH (https://parivesh.nic.in), MoEFCC's environmental clearance site (https://environmentclearance.nic.in), and other verified Indian state or national EC databases.

              Input parameters:
             - Product name: "{product_name if product_name else 'Not specified'}"
             - CAS number: "{cas_number if cas_number else 'Not specified'}"

             Instructions:
             1. Search the entire internet (including state and central EC repositories) for environmental clearances mentioning the above product name or CAS number.
             2. Extract results into a structured table with the following columns:
             - Company Name
             - Capacity (in MT or TPA)
             - Date of EC approval
             - Location (state/district)
             - Context (brief description or mention of the product within the EC)
             - Clickable EC Link (direct URL to EC PDF or project page)
             3. Ensure links are active and verifiable (prefer PARIVESH and MoEFCC sources).
             4. Output the results in Markdown table format.
             5. If no results are found, clearly state that and suggest alternative search terms or related chemicals.
             6. Keep the tone factual, structured, and research-oriented.

             Please provide a comprehensive search of all available EC databases."""

                        # Make API request to Perplexity
                        headers = {
                            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        
                        payload = {
                            "model": "sonar",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a precise environmental compliance research assistant. Provide structured, factual data from verified government sources only."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            "temperature": 0.2,
                            "max_tokens": 4000,
                            "return_citations": True,
                            "search_domain_filter": ["parivesh.nic.in", "environmentclearance.nic.in"]
                        }
                        
                        response = requests.post(
                            "https://api.perplexity.ai/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=120
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Extract the response content
                            if 'choices' in result and len(result['choices']) > 0:
                                ec_results = result['choices'][0]['message']['content']
                                
                                # Display results
                                st.success("✅ Search completed successfully!")
                                
                                # Show search parameters
                                st.markdown("---")
                                st.markdown("### 📋 Search Parameters")
                                params_col1, params_col2 = st.columns(2)
                                with params_col1:
                                    st.info(f"**Product:** {product_name if product_name else 'N/A'}")
                                with params_col2:
                                    st.info(f"**CAS Number:** {cas_number if cas_number else 'N/A'}")
                                
                                # Display results
                                st.markdown("---")
                                st.markdown("### 📊 Environmental Clearance Results")
                                st.markdown(ec_results)
                                
                                # Show citations if available
                                if 'citations' in result and result['citations']:
                                    st.markdown("---")
                                    st.markdown("### 🔗 Sources")
                                    for idx, citation in enumerate(result['citations'], 1):
                                        st.markdown(f"{idx}. [{citation}]({citation})")
                                
                                # Download option
                                st.markdown("---")
                                st.download_button(
                                    label="📥 Download Results as Text",
                                    data=ec_results,
                                    file_name=f"EC_Search_{product_name.replace(' ', '_') if product_name else cas_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain"
                                )
                                
                            else:
                                st.error("❌ No results returned from the API.")
                        
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                            st.error(f"Response: {response.text}")
                    
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. The search is taking longer than expected. Please try again.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"🔴 Network Error: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        
                    st.info("💡 **Pro Tip:** Keep saving your datasets in Tab 1 to build a comprehensive historical database for advanced analytics!")


with tab7:
    st.markdown("""
    ### 📄 Generate Comprehensive Project Report
    Create a detailed HTML report of your entire analysis including value chains, analytics, EXIM data, and market estimation.
    """)
    
    # Helper function to convert plotly figure to HTML
    def fig_to_html(fig):
        """Convert plotly figure to HTML string"""
        try:
            return fig.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})
        except Exception as e:
            st.warning(f"Could not convert chart to HTML: {str(e)}")
            return None
    
    # Helper function to capture value chain as HTML
    def capture_value_chain_html():
        """Generate HTML representation of value chain"""
        if not st.session_state.value_chain_nodes:
            return None
        
        nodes_html = ""
        for node in st.session_state.value_chain_nodes:
            if not node.get('is_group', False):
                nodes_html += f"""
                <div style='display: inline-block; margin: 10px; padding: 15px; 
                            background: {node['color']}; color: white; border-radius: 8px;
                            font-weight: 600; min-width: 120px; text-align: center;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.2);'>
                    {node['label']}<br>
                    <small style='font-size: 0.8em;'>({node['type']})</small>
                </div>
                """
        
        for node in st.session_state.value_chain_nodes:
            if node.get('is_group', False):
                category_name = node['label'].split('(')[0].strip()
                nodes_html += f"""
                <div style='display: inline-block; margin: 10px; padding: 15px; 
                            background: {node['color']}; color: white; border-radius: 8px;
                            font-weight: 600; min-width: 150px; text-align: center;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.2);'>
                    {category_name}<br>
                    <small style='font-size: 0.8em;'>(Application Category)</small>
                </div>
                """
        
        return f"""
        <div style='background: #f8fafc; padding: 20px; border-radius: 12px; 
                    border: 2px solid #cbd5e1; text-align: center;'>
            <h4 style='color: #0a2463; margin-bottom: 15px;'>Value Chain Visualization</h4>
            {nodes_html}
            <p style='margin-top: 15px; color: #64748b; font-size: 0.9em;'>
                Total Nodes: {len(st.session_state.value_chain_nodes)} | 
                Connections: {len(st.session_state.value_chain_edges)}
            </p>
        </div>
        """
    
    # Report configuration
    st.markdown("---")
    st.markdown("#### ⚙️ Report Configuration")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        report_title = st.text_input(
            "Report Title",
            value="Trade Data Analysis Report",
            help="Enter a custom title for your report"
        )
        
        company_name = st.text_input(
            "Company/Organization Name",
            value="Aarti Industries Limited",
            help="Your company or organization name"
        )
    
    with col_config2:
        report_author = st.text_input(
            "Prepared By",
            placeholder="Enter your name",
            help="Report author name"
        )
        
        report_date = st.date_input(
            "Report Date",
            value=datetime.now().date(),
            help="Date for the report"
        )
    
    # Section selection
    st.markdown("---")
    st.markdown("#### 📑 Select Sections to Include")
    
    col_section1, col_section2 = st.columns(2)
    
    with col_section1:
        include_executive_summary = st.checkbox("Executive Summary", value=True)
        include_value_chain = st.checkbox("Value Chain Analysis", value=True)
        include_analytics = st.checkbox("Analytics & Insights", value=True)
    
    with col_section2:
        include_exim = st.checkbox("EXIM Analysis", value=True)
        include_market_estimation = st.checkbox("Market Estimation", value=True)
        include_ec_analysis = st.checkbox("Environmental Clearance Analysis", value=False)
    
    # Generate report button
    st.markdown("---")
    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing report generation...")
            progress_bar.progress(10)
            
            # Pre-generate all visualizations
            charts_dict = {}
            
            status_text.text("📊 Generating analytics charts...")
            progress_bar.progress(20)
            
            # 1. Generate quarterly chart if analytics data available
            if include_analytics and st.session_state.saved_datasets:
                try:
                    first_dataset_name = list(st.session_state.saved_datasets.keys())[0]
                    dataset_info = st.session_state.saved_datasets[first_dataset_name]
                    df_processed, error = process_dataset_for_analytics(dataset_info['data'])
                    
                    if df_processed is not None and not error:
                        # Quarterly chart
                        quarterly = df_processed.groupby('Quarter').agg({
                            'Quantity_MT': 'sum',
                            'Price_USD_per_kg': lambda x: (df_processed.loc[x.index, 'Quantity_MT'] * df_processed.loc[x.index, 'Price_USD_per_kg']).sum() / df_processed.loc[x.index, 'Quantity_MT'].sum()
                        }).reset_index().sort_values('Quarter')
                        
                        fig_quarterly = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig_quarterly.add_trace(
                            go.Bar(
                                x=quarterly['Quarter'],
                                y=quarterly['Quantity_MT'],
                                name="Volume (MT)",
                                marker_color='#fb923c',
                            ),
                            secondary_y=False
                        )
                        
                        fig_quarterly.add_trace(
                            go.Scatter(
                                x=quarterly['Quarter'],
                                y=quarterly['Price_USD_per_kg'],
                                name="Weighted Avg Price ($/kg)",
                                mode='lines+markers',
                                line=dict(color='#10b981', width=3),
                                marker=dict(size=10),
                            ),
                            secondary_y=True
                        )
                        
                        fig_quarterly.update_layout(
                            title="Quarterly Volume & Weighted Average Price",
                            height=500,
                            showlegend=True,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                        )
                        
                        fig_quarterly.update_xaxes(title_text="Quarter", tickangle=45)
                        fig_quarterly.update_yaxes(title_text="Volume (MT)", secondary_y=False)
                        fig_quarterly.update_yaxes(title_text="Price ($/kg)", secondary_y=True)
                        
                        charts_dict['quarterly'] = fig_quarterly
                        
                        # Geographic chart
                        country_col = df_processed.attrs.get('country_col')
                        if country_col and country_col in df_processed.columns:
                            geo_data = df_processed.groupby(country_col).agg({
                                'Quantity_MT': 'sum',
                                'Price_USD_per_kg': lambda x: (df_processed.loc[x.index, 'Quantity_MT'] * df_processed.loc[x.index, 'Price_USD_per_kg']).sum() / df_processed.loc[x.index, 'Quantity_MT'].sum()
                            }).reset_index()
                            
                            geo_data.columns = ['Country', 'Volume_MT', 'Avg_Price']
                            
                            fig_geo = px.choropleth(
                                geo_data,
                                locations='Country',
                                locationmode='country names',
                                color='Volume_MT',
                                hover_name='Country',
                                hover_data={'Volume_MT': ':.2f', 'Avg_Price': ':.2f'},
                                color_continuous_scale='Oranges',
                                title="Volume Distribution by Country (MT)"
                            )
                            
                            fig_geo.update_layout(
                                height=500,
                                geo=dict(
                                    showframe=False,
                                    showcoastlines=True,
                                    projection_type='natural earth',
                                ),
                                paper_bgcolor='white',
                            )
                            
                            charts_dict['geographic'] = fig_geo
                        
                        # Top countries bar chart
                        if country_col and country_col in df_processed.columns:
                            top_countries = df_processed.groupby(country_col).agg({
                                'Quantity_MT': 'sum'
                            }).reset_index().sort_values('Quantity_MT', ascending=False).head(10)
                            
                            fig_top_countries = go.Figure(data=[
                                go.Bar(
                                    x=top_countries['Quantity_MT'],
                                    y=top_countries[country_col],
                                    orientation='h',
                                    marker_color='#3b82f6',
                                )
                            ])
                            
                            fig_top_countries.update_layout(
                                title="Top 10 Countries by Volume",
                                xaxis_title="Volume (MT)",
                                yaxis_title="Country",
                                height=500,
                                paper_bgcolor='white',
                                plot_bgcolor='white',
                            )
                            
                            charts_dict['top_countries'] = fig_top_countries
                        
                        # Market estimation pie chart
                        if include_market_estimation and st.session_state.downstreams:
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=[ds['name'] for ds in st.session_state.downstreams],
                                values=[ds['calculated_demand'] for ds in st.session_state.downstreams],
                                hole=0.4,
                                marker=dict(
                                    colors=['#fbbf24', '#22c55e', '#8b5cf6', '#ef4444', '#3b82f6', '#ec4899'],
                                ),
                                textposition='auto',
                                textinfo='label+percent',
                            )])
                            
                            fig_pie.update_layout(
                                title="Market Distribution by Downstream Application",
                                height=500,
                                paper_bgcolor='white',
                                showlegend=True,
                            )
                            
                            charts_dict['market_pie'] = fig_pie
                            
                            # Category breakdown bar chart
                            category_totals = {}
                            for ds in st.session_state.downstreams:
                                cat = ds['category']
                                if cat not in category_totals:
                                    category_totals[cat] = 0
                                category_totals[cat] += ds['calculated_demand']
                            
                            fig_category = go.Figure(data=[
                                go.Bar(
                                    x=list(category_totals.keys()),
                                    y=list(category_totals.values()),
                                    marker_color=['#fbbf24', '#22c55e', '#8b5cf6'],
                                )
                            ])
                            
                            fig_category.update_layout(
                                title="Market Demand by Category",
                                xaxis_title="Category",
                                yaxis_title="Demand (MT)",
                                height=400,
                                paper_bgcolor='white',
                                plot_bgcolor='white',
                            )
                            
                            charts_dict['category'] = fig_category
                
                except Exception as e:
                    st.warning(f"Could not generate some charts: {str(e)}")
            
            status_text.text("📝 Building HTML report...")
            progress_bar.progress(40)
            
            # Build HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{report_title}</title>
                <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
                    
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    
                    body {{
                        font-family: 'Poppins', Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        background: #ffffff;
                        padding: 40px;
                        max-width: 1400px;
                        margin: 0 auto;
                    }}
                    
                    .cover-page {{
                        text-align: center;
                        padding: 100px 0;
                        page-break-after: always;
                        border-bottom: 4px solid #f97316;
                        min-height: 80vh;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    }}
                    
                    .cover-page h1 {{
                        font-size: 3rem;
                        color: #0a2463;
                        margin-bottom: 20px;
                        font-weight: 700;
                    }}
                    
                    .cover-page .company {{
                        font-size: 1.8rem;
                        color: #f97316;
                        margin-bottom: 40px;
                        font-weight: 600;
                    }}
                    
                    .cover-page .metadata {{
                        margin-top: 60px;
                        font-size: 1.1rem;
                        color: #64748b;
                    }}
                    
                    h2 {{
                        color: #0a2463;
                        font-size: 2rem;
                        margin-top: 40px;
                        margin-bottom: 20px;
                        border-bottom: 3px solid #f97316;
                        padding-bottom: 10px;
                        page-break-after: avoid;
                    }}
                    
                    h3 {{
                        color: #1e3a8a;
                        font-size: 1.5rem;
                        margin-top: 30px;
                        margin-bottom: 15px;
                        page-break-after: avoid;
                    }}
                    
                    h4 {{
                        color: #475569;
                        font-size: 1.2rem;
                        margin-top: 20px;
                        margin-bottom: 10px;
                    }}
                    
                    p {{
                        margin-bottom: 15px;
                        text-align: justify;
                    }}
                    
                    .section {{
                        margin-bottom: 40px;
                        page-break-inside: avoid;
                    }}
                    
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                        font-size: 0.9rem;
                        page-break-inside: avoid;
                    }}
                    
                    table thead {{
                        background: linear-gradient(135deg, #0a2463 0%, #1e3a8a 100%);
                        color: white;
                    }}
                    
                    table th {{
                        padding: 12px;
                        text-align: left;
                        font-weight: 600;
                        border: 1px solid #cbd5e1;
                    }}
                    
                    table td {{
                        padding: 10px;
                        border: 1px solid #cbd5e1;
                    }}
                    
                    table tbody tr:nth-child(even) {{
                        background-color: #f8fafc;
                    }}
                    
                    table tbody tr:hover {{
                        background-color: #fef3c7;
                    }}
                    
                    .metric-box {{
                        display: inline-block;
                        padding: 20px;
                        margin: 10px;
                        border-radius: 10px;
                        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                        border: 2px solid #f97316;
                        min-width: 200px;
                        text-align: center;
                    }}
                    
                    .metric-box .value {{
                        font-size: 2rem;
                        font-weight: 700;
                        color: #f97316;
                        display: block;
                    }}
                    
                    .metric-box .label {{
                        font-size: 0.9rem;
                        color: #0a2463;
                        font-weight: 600;
                        text-transform: uppercase;
                        margin-top: 5px;
                    }}
                    
                    .info-box {{
                        background: #e0f2fe;
                        border-left: 4px solid #0284c7;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 5px;
                    }}
                    
                    .warning-box {{
                        background: #fef3c7;
                        border-left: 4px solid #f59e0b;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 5px;
                    }}
                    
                    .success-box {{
                        background: #d1fae5;
                        border-left: 4px solid #10b981;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 5px;
                    }}
                    
                    .page-break {{
                        page-break-after: always;
                    }}
                    
                    ul, ol {{
                        margin-left: 30px;
                        margin-bottom: 15px;
                    }}
                    
                    li {{
                        margin-bottom: 8px;
                    }}
                    
                    .footer {{
                        margin-top: 60px;
                        padding-top: 20px;
                        border-top: 2px solid #cbd5e1;
                        text-align: center;
                        color: #64748b;
                        font-size: 0.9rem;
                    }}
                    
                    .chart-container {{
                        background: #f8fafc;
                        border: 2px solid #cbd5e1;
                        padding: 20px;
                        margin: 20px 0;
                        border-radius: 10px;
                        min-height: 500px;
                    }}
                    
                    @media print {{
                        body {{
                            padding: 20px;
                        }}
                        
                        .no-print {{
                            display: none;
                        }}
                        
                        h2, h3 {{
                            page-break-after: avoid;
                        }}
                        
                        table {{
                            page-break-inside: avoid;
                        }}
                        
                        .chart-container {{
                            page-break-inside: avoid;
                        }}
                    }}
                </style>
            </head>
            <body>
                <!-- Cover Page -->
                <div class="cover-page">
                    <h1>{report_title}</h1>
                    <div class="company">{company_name}</div>
                    <div class="metadata">
                        <p><strong>Prepared By:</strong> {report_author if report_author else 'N/A'}</p>
                        <p><strong>Report Date:</strong> {report_date.strftime('%B %d, %Y')}</p>
                        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            """
            
            progress_bar.progress(50)
            
            # Table of Contents
            html_content += """
                <div class="section">
                    <h2>📑 Table of Contents</h2>
                    <ol>
            """
            
            toc_items = []
            section_num = 1
            
            if include_executive_summary:
                toc_items.append(f"{section_num}. Executive Summary")
                section_num += 1
            if include_value_chain:
                toc_items.append(f"{section_num}. Value Chain Analysis")
                section_num += 1
            if include_analytics:
                toc_items.append(f"{section_num}. Analytics & Insights")
                section_num += 1
            if include_exim:
                toc_items.append(f"{section_num}. EXIM Analysis")
                section_num += 1
            if include_market_estimation:
                toc_items.append(f"{section_num}. Market Estimation")
                section_num += 1
            if include_ec_analysis:
                toc_items.append(f"{section_num}. Environmental Clearance Analysis")
                section_num += 1
            
            for item in toc_items:
                html_content += f"<li>{item}</li>"
            
            html_content += """
                    </ol>
                </div>
                <div class="page-break"></div>
            """
            
            # Section counter
            current_section = 1
            
            status_text.text("📊 Adding Executive Summary...")
            progress_bar.progress(55)
            
            # Executive Summary
            if include_executive_summary:
                html_content += f"""
                <div class="section">
                    <h2>{current_section}. Executive Summary</h2>
                    <p>This report presents a comprehensive analysis of trade data processed through the Aarti Industries Trade Data Standardization System. The analysis covers multiple dimensions including value chain mapping, market analytics, EXIM analysis, and demand estimation.</p>
                """
                current_section += 1
                
                # Key metrics
                total_datasets = len(st.session_state.saved_datasets)
                total_value_chain_nodes = len([n for n in st.session_state.value_chain_nodes if not n.get('is_group', False)])
                total_downstreams = len(st.session_state.downstreams)
                
                html_content += f"""
                    <h3>Key Highlights</h3>
                    <div class="metric-box">
                        <span class="value">{total_datasets}</span>
                        <span class="label">Datasets Processed</span>
                    </div>
                    <div class="metric-box">
                        <span class="value">{total_value_chain_nodes}</span>
                        <span class="label">Value Chain Nodes</span>
                    </div>
                    <div class="metric-box">
                        <span class="value">{total_downstreams}</span>
                        <span class="label">Downstream Applications</span>
                    </div>
                """
                
                if st.session_state.saved_datasets:
                    total_rows = sum([ds['rows'] for ds in st.session_state.saved_datasets.values()])
                    html_content += f"""
                    <div class="metric-box">
                        <span class="value">{total_rows:,}</span>
                        <span class="label">Total Data Rows</span>
                    </div>
                    """
                
                html_content += """
                </div>
                <div class="page-break"></div>
                """
            
            status_text.text("🔗 Adding Value Chain Analysis...")
            progress_bar.progress(60)
            
            # Value Chain Analysis
            if include_value_chain and st.session_state.value_chain_nodes:
                html_content += f"""
                <div class="section">
                    <h2>{current_section}. Value Chain Analysis</h2>
                    <p>The value chain represents the relationship between raw materials, intermediates, target products, and downstream applications.</p>
                    
                    <h3>Value Chain Visualization</h3>
                """
                current_section += 1
                
                # Add value chain visualization
                value_chain_html = capture_value_chain_html()
                if value_chain_html:
                    html_content += value_chain_html
                
                html_content += """
                    <h3>Value Chain Components</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Molecule Name</th>
                                <th>Type</th>
                                <th>Category</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for node in st.session_state.value_chain_nodes:
                    if not node.get('is_group', False):
                        html_content += f"""
                            <tr>
                                <td><strong>{node['label']}</strong></td>
                                <td>{node['type']}</td>
                                <td>Chemical Molecule</td>
                            </tr>
                        """
                
                html_content += """
                        </tbody>
                    </table>
                """
                
                # Group nodes
                group_nodes = [node for node in st.session_state.value_chain_nodes if node.get('is_group', False)]
                if group_nodes:
                    html_content += """
                    <h3>Downstream Application Categories</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>Number of Applications</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    
                    for node in group_nodes:
                        category_name = node['label'].split('(')[0].strip()
                        count_str = node['label'].split('(')[1].split(' ')[0] if '(' in node['label'] else '0'
                        html_content += f"""
                            <tr>
                                <td><strong>{category_name}</strong></td>
                                <td>{count_str}</td>
                            </tr>
                        """
                    
                    html_content += """
                        </tbody>
                    </table>
                    """
                
                html_content += """
                </div>
                <div class="page-break"></div>
                """
            
            status_text.text("📈 Adding Analytics & Insights...")
            progress_bar.progress(65)
            
            # Analytics & Insights with interactive charts
            if include_analytics and st.session_state.saved_datasets:
                html_content += f"""
                <div class="section">
                    <h2>{current_section}. Analytics & Insights</h2>
                    <p>Comprehensive analytics derived from processed trade data with interactive visualizations.</p>
                """
                current_section += 1
                
                try:
                    first_dataset_name = list(st.session_state.saved_datasets.keys())[0]
                    dataset_info = st.session_state.saved_datasets[first_dataset_name]
                    df_processed, error = process_dataset_for_analytics(dataset_info['data'])
                    
                    if df_processed is not None and not error:
                        # Add FY column
                        def get_fy_from_quarter(quarter):
                            if pd.isna(quarter):
                                return None
                            try:
                                return quarter.split()[0]
                            except:
                                return None
                        
                        df_processed['FY'] = df_processed['Quarter'].apply(get_fy_from_quarter)
                        
                        # Key Metrics
                        total_volume = df_processed['Quantity_MT'].sum()
                        avg_price = (df_processed['Quantity_MT'] * df_processed['Price_USD_per_kg']).sum() / total_volume if total_volume > 0 else 0
                        unique_quarters = df_processed['Quarter'].nunique()
                        
                        html_content += f"""
                        <h3>Overview Metrics - {first_dataset_name}</h3>
                        <div class="metric-box">
                            <span class="value">{total_volume:,.2f}</span>
                            <span class="label">Total Volume (MT)</span>
                        </div>
                        <div class="metric-box">
                            <span class="value">${avg_price:.2f}</span>
                            <span class="label">Weighted Avg Price/kg</span>
                        </div>
                        <div class="metric-box">
                            <span class="value">{unique_quarters}</span>
                            <span class="label">Time Periods</span>
                        </div>
                        """
                        
                        # Add quarterly chart
                        if 'quarterly' in charts_dict:
                            chart_html = fig_to_html(charts_dict['quarterly'])
                            if chart_html:
                                html_content += f"""
                                <h3>Quarterly Volume & Price Trends</h3>
                                <div class="chart-container">
                                    {chart_html}
                                </div>
                                """
                        
                        # Quarterly table
                        quarterly = df_processed.groupby('Quarter').agg({
                            'Quantity_MT': 'sum',
                            'Price_USD_per_kg': lambda x: (df_processed.loc[x.index, 'Quantity_MT'] * df_processed.loc[x.index, 'Price_USD_per_kg']).sum() / df_processed.loc[x.index, 'Quantity_MT'].sum()
                        }).reset_index().sort_values('Quarter')
                        
                        html_content += """
                        <h3>Quarterly Data Table</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Quarter</th>
                                    <th>Volume (MT)</th>
                                    <th>Weighted Avg Price ($/kg)</th>
                                </tr>
                            </thead>
                            <tbody>
                        """
                        
                        for _, row in quarterly.iterrows():
                            html_content += f"""
                                <tr>
                                    <td><strong>{row['Quarter']}</strong></td>
                                    <td>{row['Quantity_MT']:,.2f}</td>
                                    <td>${row['Price_USD_per_kg']:.2f}</td>
                                </tr>
                            """
                        
                        html_content += """
                            </tbody>
                        </table>
                        """
                        
                        # Geographic analysis
                        country_col = df_processed.attrs.get('country_col')
                        if country_col and country_col in df_processed.columns:
                            # Add geographic chart
                            if 'geographic' in charts_dict:
                                chart_html = fig_to_html(charts_dict['geographic'])
                                if chart_html:
                                    html_content += f"""
                                    <h3>Geographic Distribution</h3>
                                    <div class="chart-container">
                                        {chart_html}
                                    </div>
                                    """
                            
                            # Add top countries bar chart
                            if 'top_countries' in charts_dict:
                                chart_html = fig_to_html(charts_dict['top_countries'])
                                if chart_html:
                                    html_content += f"""
                                    <h3>Top 10 Countries by Volume</h3>
                                    <div class="chart-container">
                                        {chart_html}
                                    </div>
                                    """
                            
                            # Geographic table
                            geo_data = df_processed.groupby(country_col).agg({
                                'Quantity_MT': 'sum',
                                'Price_USD_per_kg': lambda x: (df_processed.loc[x.index, 'Quantity_MT'] * df_processed.loc[x.index, 'Price_USD_per_kg']).sum() / df_processed.loc[x.index, 'Quantity_MT'].sum()
                            }).reset_index().sort_values('Quantity_MT', ascending=False).head(15)
                            
                            geo_data.columns = ['Country', 'Volume_MT', 'Avg_Price']
                            
                            html_content += """
                            <h3>Top 15 Countries by Volume</h3>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Rank</th>
                                        <th>Country</th>
                                        <th>Volume (MT)</th>
                                        <th>Avg Price ($/kg)</th>
                                        <th>Market Share</th>
                                    </tr>
                                </thead>
                                <tbody>
                            """
                            
                            for idx, (_, row) in enumerate(geo_data.iterrows(), 1):
                                market_share = (row['Volume_MT'] / total_volume * 100) if total_volume > 0 else 0
                                html_content += f"""
                                    <tr>
                                        <td><strong>{idx}</strong></td>
                                        <td>{row['Country']}</td>
                                        <td>{row['Volume_MT']:,.2f}</td>
                                        <td>${row['Avg_Price']:.2f}</td>
                                        <td>{market_share:.1f}%</td>
                                    </tr>
                                """
                            
                            html_content += """
                                </tbody>
                            </table>
                            """
                
                except Exception as e:
                    html_content += f"""
                    <div class="warning-box">
                        <strong>Note:</strong> Could not generate detailed analytics. Error: {str(e)}
                    </div>
                    """
                
                html_content += """
                </div>
                <div class="page-break"></div>
                """
            
            status_text.text("📦 Adding EXIM Analysis...")
            progress_bar.progress(70)
            
            # EXIM Analysis with tables
            if include_exim and st.session_state.saved_datasets:
                html_content += f"""
                <div class="section">
                    <h2>{current_section}. EXIM Analysis</h2>
                    <p>Detailed Import, Export, and Global trade analysis with financial year breakdowns.</p>
                    
                    <h3>Calculation Methodology</h3>
                    <div class="info-box">
                        <ul>
                            <li><strong>Unit Filter:</strong> Only METRIC TON and KILOGRAMS rows are included</li>
                            <li><strong>Conversion:</strong> MT → kg (×1000) | Price/MT → Price/kg (÷1000)</li>
                            <li><strong>Formula:</strong> Weighted Avg Price ($/kg) = Σ(qty_kg × price_$/kg) ÷ Σ(qty_kg)</li>
                            <li><strong>Financial Year:</strong> April to March cycle</li>
                        </ul>
                    </div>
                """
                current_section += 1
                
                # Process EXIM data
                try:
                    financial_years = [
                        "FY 2020-21", "FY 2021-22", "FY 2022-23", 
                        "FY 2023-24", "FY 2024-25", "FY 2025-26"
                    ]
                    
                    # Process EXIM data
                    exim_data = {
                        'Imports': [],
                        'Exports': [],
                        'Global': []
                    }
                    
                    def get_financial_year(date_val):
                        try:
                            if pd.isna(date_val):
                                return None
                            if isinstance(date_val, str):
                                date_obj = pd.to_datetime(date_val, dayfirst=True, errors='coerce')
                            else:
                                date_obj = pd.to_datetime(date_val)
                            if pd.isna(date_obj):
                                return None
                            year = date_obj.year
                            month = date_obj.month
                            if month >= 4:
                                return f"FY {year}-{str(year+1)[2:]}"
                            else:
                                return f"FY {year-1}-{str(year)[2:]}"
                        except:
                            return None
                    
                    # Process each dataset
                    for dataset_name, dataset_info in st.session_state.saved_datasets.items():
                        data_type = dataset_info['type']
                        df = dataset_info['data'].copy()
                        
                        section = 'Imports' if data_type == 'Import' else ('Exports' if data_type == 'Export' else 'Global')
                        
                        if 'Standardized Name' not in df.columns:
                            continue
                        
                        # Find columns
                        date_col = next((col for col in df.columns if 'date' in col.lower() or 'period' in col.lower()), None)
                        qty_col = next((col for col in df.columns if 'quantity' in col.lower() or 'qty' in col.lower()), None)
                        unit_col = next((col for col in df.columns if 'unit' in col.lower() or 'uqc' in col.lower()), None)
                        value_col = next((col for col in df.columns if 'value' in col.lower() and 'usd' in col.lower()), None)
                        
                        if not all([date_col, qty_col, unit_col, value_col]):
                            continue
                        
                        df['FY'] = df[date_col].apply(get_financial_year)
                        
                        # Get unique products
                        products = df['Standardized Name'].unique()
                        
                        for product in products:
                            product_df = df[df['Standardized Name'] == product].copy()
                            
                            row = {
                                'Product': f"{product} — [{dataset_name}]",
                                'Relation': st.session_state.product_relations.get(f"{product}_{dataset_name}", "Product")
                            }
                            
                            for fy in financial_years:
                                fy_data = product_df[product_df['FY'] == fy].copy()
                                
                                if not fy_data.empty:
                                    # Filter MT/KG only
                                    fy_data['Unit_Upper'] = fy_data[unit_col].astype(str).str.upper()
                                    valid_data = fy_data[fy_data['Unit_Upper'].str.contains('METRIC TON|KILOGRAM', regex=True, na=False)].copy()
                                    
                                    if not valid_data.empty:
                                        total_qty_mt = valid_data[qty_col].sum()
                                        total_value_usd = valid_data[value_col].sum()
                                        
                                        # Calculate weighted avg price
                                        avg_price = (total_value_usd / (total_qty_mt * 1000)) if total_qty_mt > 0 else 0
                                        
                                        row[f'{fy}_Qty'] = round(total_qty_mt, 2)
                                        row[f'{fy}_Price'] = round(avg_price, 2)
                                    else:
                                        row[f'{fy}_Qty'] = None
                                        row[f'{fy}_Price'] = None
                                else:
                                    row[f'{fy}_Qty'] = None
                                    row[f'{fy}_Price'] = None
                            
                            exim_data[section].append(row)
                    
                    # Generate tables for each section
                    for section in ['Imports', 'Exports', 'Global']:
                        if exim_data[section]:
                            html_content += f"""
                            <h3>{section}</h3>
                            <table>
                                <thead>
                                    <tr>
                                        <th rowspan="2">Product</th>
                                        <th rowspan="2">Relation</th>
                            """
                            
                            for fy in financial_years:
                                html_content += f'<th colspan="2" style="text-align: center;">{fy}</th>'
                            
                            html_content += """
                                    </tr>
                                    <tr>
                            """
                            
                            for fy in financial_years:
                                html_content += '<th>Qty (MT)</th><th>Price ($/kg)</th>'
                            
                            html_content += """
                                    </tr>
                                </thead>
                                <tbody>
                            """
                            
                            for row_data in exim_data[section][:25]:  # Limit to first 25 products
                                html_content += f"""
                                    <tr>
                                        <td><strong>{row_data['Product']}</strong></td>
                                        <td>{row_data['Relation']}</td>
                                """
                                
                                for fy in financial_years:
                                    qty = row_data.get(f'{fy}_Qty')
                                    price = row_data.get(f'{fy}_Price')
                                    
                                    qty_str = f"{qty:,.2f}" if qty else "—"
                                    price_str = f"${price:.2f}" if price else "—"
                                    
                                    html_content += f"""
                                        <td>{qty_str}</td>
                                        <td>{price_str}</td>
                                    """
                                
                                html_content += "</tr>"
                            
                            html_content += """
                                </tbody>
                            </table>
                            """
                            
                            if len(exim_data[section]) > 25:
                                html_content += f"""
                                <div class="info-box">
                                    <strong>Note:</strong> Showing top 25 products. Total products: {len(exim_data[section])}
                                </div>
                                """
                    
                except Exception as e:
                    html_content += f"""
                    <div class="warning-box">
                        <strong>Note:</strong> Could not generate EXIM tables. Error: {str(e)}
                    </div>
                    """
                
                html_content += """
                </div>
                <div class="page-break"></div>
                """
            
            status_text.text("💰 Adding Market Estimation...")
            progress_bar.progress(80)
            
            # Market Estimation with interactive charts
            if include_market_estimation and st.session_state.downstreams:
                html_content += f"""
                <div class="section">
                    <h2>{current_section}. Market Estimation</h2>
                    <p>Comprehensive market demand estimation based on downstream application analysis.</p>
                """
                current_section += 1
                
                # Calculate totals
                pharma_demand = sum([ds['calculated_demand'] for ds in st.session_state.downstreams if ds['category'] == 'Pharma'])
                agro_demand = sum([ds['calculated_demand'] for ds in st.session_state.downstreams if ds['category'] == 'Agro'])
                others_demand = sum([ds['calculated_demand'] for ds in st.session_state.downstreams if ds['category'] == 'Others'])
                total_demand = pharma_demand + agro_demand + others_demand
                
                html_content += f"""
                    <h3>Total Market Demand</h3>
                    <div class="metric-box">
                        <span class="value">{total_demand:,.2f} MT</span>
                        <span class="label">Total Estimated Demand</span>
                    </div>
                    
                    <h3>Category Breakdown</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>Demand (MT)</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Pharmaceuticals</strong></td>
                                <td>{pharma_demand:,.2f}</td>
                                <td>{(pharma_demand/total_demand*100) if total_demand > 0 else 0:.1f}%</td>
                            </tr>
                            <tr>
                                <td><strong>Agrochemicals</strong></td>
                                <td>{agro_demand:,.2f}</td>
                                <td>{(agro_demand/total_demand*100) if total_demand > 0 else 0:.1f}%</td>
                            </tr>
                            <tr>
                                <td><strong>Others</strong></td>
                                <td>{others_demand:,.2f}</td>
                                <td>{(others_demand/total_demand*100) if total_demand > 0 else 0:.1f}%</td>
                            </tr>
                        </tbody>
                    </table>
                """
                
                # Add market pie chart
                if 'market_pie' in charts_dict:
                    chart_html = fig_to_html(charts_dict['market_pie'])
                    if chart_html:
                        html_content += f"""
                        <h3>Market Distribution Visualization</h3>
                        <div class="chart-container">
                            {chart_html}
                        </div>
                        """
                
                # Add category bar chart
                if 'category' in charts_dict:
                    chart_html = fig_to_html(charts_dict['category'])
                    if chart_html:
                        html_content += f"""
                        <h3>Category-wise Demand</h3>
                        <div class="chart-container">
                            {chart_html}
                        </div>
                        """
                
                # Downstream details
                html_content += """
                    <h3>Downstream Applications Detail</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Downstream Product</th>
                                <th>Category</th>
                                <th>Demand (MT)</th>
                                <th>Norm</th>
                                <th>Calculated (MT)</th>
                                <th>Contribution %</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for ds in st.session_state.downstreams:
                    contribution_pct = (ds['calculated_demand'] / total_demand * 100) if total_demand > 0 else 0
                    html_content += f"""
                            <tr>
                                <td><strong>{ds['name']}</strong></td>
                                <td>{ds['category']}</td>
                                <td>{ds['demand_mt']:,.2f}</td>
                                <td>{ds['norm']:.4f}</td>
                                <td>{ds['calculated_demand']:,.2f}</td>
                                <td>{contribution_pct:.1f}%</td>
                            </tr>
                    """
                
                html_content += """
                        </tbody>
                    </table>
                </div>
                <div class="page-break"></div>
                """
            
            status_text.text("🌱 Adding Environmental Clearance Analysis...")
            progress_bar.progress(90)
            
            # EC Analysis
            if include_ec_analysis:
                html_content += f"""
                <div class="section">
                    <h2>{current_section}. Environmental Clearance Analysis</h2>
                    <p>Environmental clearance information for analyzed chemical products.</p>
                    
                    <div class="warning-box">
                        <strong>Note:</strong> EC data is searched on-demand through the application. 
                        Results include company names, capacities, approval dates, locations, and document links.
                    </div>
                    
                    <h3>Data Sources</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Source</th>
                                <th>Description</th>
                                <th>URL</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>PARIVESH Portal</strong></td>
                                <td>Environmental clearance information system</td>
                                <td><a href="https://parivesh.nic.in" target="_blank">parivesh.nic.in</a></td>
                            </tr>
                            <tr>
                                <td><strong>MoEFCC</strong></td>
                                <td>Ministry of Environment, Forest and Climate Change</td>
                                <td><a href="https://environmentclearance.nic.in" target="_blank">environmentclearance.nic.in</a></td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <h3>How to Access EC Data</h3>
                    <ol>
                        <li>Navigate to Tab 6 - Environmental Clearance Analysis</li>
                        <li>Enter the chemical product name in the search box</li>
                        <li>Click "Search EC Data" to fetch information from government portals</li>
                        <li>Review the results including company names, capacities, and approval dates</li>
                        <li>Download individual EC documents from the provided links</li>
                    </ol>
                </div>
                <div class="page-break"></div>
                """
                current_section += 1
            
            status_text.text("✅ Finalizing report...")
            progress_bar.progress(95)
            
            # Footer
            html_content += f"""
                <div class="footer">
                    <p><strong>Generated by Aarti Industries Trade Data Standardization System</strong></p>
                    <p>Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>&copy; {datetime.now().year} {company_name}. All rights reserved.</p>
                </div>
            </body>
            </html>
            """
            
            progress_bar.progress(100)
            status_text.text("✅ Report generated successfully!")
            
            # Display success
            st.success("✅ Report generated successfully with interactive charts!")
            
            # Preview
            st.markdown("---")
            st.markdown("### 👁️ Report Preview")
            with st.expander("View HTML Report", expanded=False):
                st.components.v1.html(html_content, height=600, scrolling=True)
            
            # Download options
            st.markdown("---")
            st.markdown("### 📥 Download Options")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                st.download_button(
                    label="📄 Download HTML Report",
                    data=html_content,
                    file_name=f"Trade_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col_dl2:
                st.info("""
                **To Save as PDF:**
                1. Download HTML
                2. Open in browser
                3. Press Ctrl+P (Cmd+P)
                4. Save as PDF
                """)
            
            # Statistics
            st.markdown("---")
            st.markdown("### 📊 Report Statistics")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                sections_included = sum([
                    include_executive_summary,
                    include_value_chain,
                    include_analytics,
                    include_exim,
                    include_market_estimation,
                    include_ec_analysis
                ])
                st.metric("Sections", sections_included)
            
            with stat_col2:
                charts_included = len(charts_dict)
                st.metric("Charts", charts_included)
            
            with stat_col3:
                st.metric("Datasets", len(st.session_state.saved_datasets))
            
            with stat_col4:
                file_size_kb = len(html_content) / 1024
                st.metric("Size", f"{file_size_kb:.1f} KB")
            
            # Tips
            st.markdown("---")
            with st.expander("💡 Tips for Best Results"):
                st.markdown("""
                **Viewing:**
                - Open HTML in Chrome, Firefox, Edge, or Safari
                - All charts are interactive and embedded using Plotly
                - Formatting is fully preserved
                - No external dependencies required
                
                **Converting to PDF:**
                - Use Print to PDF (Ctrl/Cmd + P)
                - Enable background graphics
                - Use A4 or Letter size
                - Set margins to "Default"
                
                **Chart Features:**
                - Charts are fully interactive in HTML
                - Hover over data points for details
                - Zoom, pan, and reset view available
                - Download charts as PNG from chart menu
                
                **Troubleshooting:**
                - If charts don't load, ensure JavaScript is enabled
                - Charts require internet connection for Plotly CDN
                - For offline use, save complete webpage (Ctrl+S)
                - Large reports may take a moment to load all charts
                """)
        
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
            st.exception(e)
        
        finally:
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
    
    # Help section
    st.markdown("---")
    with st.expander("❓ How to Generate Reports", expanded=False):
        st.markdown("""
        ### Complete Report Generation Guide
        
        **Prerequisites:**
        1. Process and save datasets in Tab 1
        2. Build value chains in Tab 3
        3. Review analytics in Tab 4
        4. Add market estimations in Tab 5
        
        **Steps:**
        1. **Configure** - Enter title, company, author, date
        2. **Select Sections** - Choose what to include
        3. **Generate** - Click button and wait
        4. **Preview** - Review in expander
        5. **Download** - Get HTML or convert to PDF
        
        **What's Included:**
        - **Value Chain**: Visual diagram with all nodes and connections
        - **Quarterly Trends**: Interactive chart showing volume & price over time
        - **Geographic Map**: Interactive world map with volume distribution
        - **Top Countries**: Bar chart of leading trade partners
        - **Market Pie**: Category breakdown visualization
        - **Category Chart**: Bar chart showing demand by category
        - **EXIM Tables**: Complete trade data tables with FY breakdown
        - **All Metrics**: Key statistics and insights
        
        **Chart Generation:**
        Charts are automatically generated from your data:
        - Quarterly analysis from Analytics tab
        - Geographic distribution from country data
        - Market charts from downstream applications
        - Value chain visualization from Tab 3
        
        **Interactive Features:**
        - Hover over charts for detailed information
        - Zoom and pan on geographic maps
        - Download individual charts as PNG
        - Fully responsive and interactive in browser
        """)
    
    # Data availability
    st.markdown("---")
    st.markdown("### 📊 Data Availability")
    
    avail_col1, avail_col2, avail_col3, avail_col4 = st.columns(4)
    
    with avail_col1:
        if st.session_state.saved_datasets:
            st.success(f"✅ {len(st.session_state.saved_datasets)} Datasets")
        else:
            st.warning("⚠️ No Datasets")
    
    with avail_col2:
        if st.session_state.value_chain_nodes:
            node_count = len([n for n in st.session_state.value_chain_nodes if not n.get('is_group', False)])
            st.success(f"✅ {node_count} Nodes")
        else:
            st.warning("⚠️ No Value Chain")
    
    with avail_col3:
        if st.session_state.downstreams:
            st.success(f"✅ {len(st.session_state.downstreams)} Apps")
        else:
            st.warning("⚠️ No Market Data")
    
    with avail_col4:
        has_exim = any(ds['type'] in ['Import', 'Export', 'Global'] 
                       for ds in st.session_state.saved_datasets.values()) if st.session_state.saved_datasets else False
        if has_exim:
            st.success("✅ EXIM Data")
        else:
            st.warning("⚠️ No EXIM")



# Add this inside tab8:
with tab8:
    st.markdown("""
    ### 🤖 Advanced Analytics & Predictions
    Use machine learning to predict prices and volumes based on historical data patterns.
    """)

    if not st.session_state.saved_datasets:
        st.info("📊 No saved datasets available. Please process and save data in the Data Processing tab first.")
    else:
        st.markdown("---")
        st.markdown("#### 🎯 Select Data for Prediction")

        # Dataset selection
        selected_dataset = st.selectbox(
            "Choose Dataset",
            options=list(st.session_state.saved_datasets.keys()),
            help="Select the cleaned dataset for prediction"
        )

        if selected_dataset:
            dataset = st.session_state.saved_datasets[selected_dataset]
            df = dataset['data'].copy()

            try:
                # Find relevant columns
                date_col = next((col for col in df.columns if 'date' in col.lower() or 'period' in col.lower()), None)
                qty_col = next((col for col in df.columns if 'quantity' in col.lower() or 'qty' in col.lower() or 'weight' in col.lower()), None)
                value_col = next((col for col in df.columns if 'value' in col.lower() or 'amount' in col.lower() or 'price' in col.lower()), None)
                
                if all([date_col, qty_col, value_col]):
                    # Prepare data with proper error handling
                    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
                    df['Year'] = df['Date'].dt.year
                    df['Month'] = df['Date'].dt.month
                    df['Quarter'] = df['Date'].dt.quarter
                    
                    # Convert to numeric and handle errors
                    df['Quantity_MT'] = pd.to_numeric(df[qty_col], errors='coerce')
                    df['Value_USD'] = pd.to_numeric(df[value_col], errors='coerce')
                    
                    # Remove rows with missing or zero quantities/values
                    df = df.dropna(subset=['Date', 'Quantity_MT', 'Value_USD'])
                    df = df[(df['Quantity_MT'] > 0) & (df['Value_USD'] > 0)]
                    
                    if len(df) < 10:
                        st.error("Insufficient valid data for prediction. Need at least 10 data points.")
                        st.stop()
                    
                    # Calculate price (now safe from division by zero)
                    df['Price_USD_per_kg'] = df['Value_USD'] / (df['Quantity_MT'] * 1000)
                    
                    # Replace any infinite values
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.dropna()
                    
                    # Group by month for time series
                    monthly_data = df.groupby([df['Date'].dt.to_period('M')]).agg({
                        'Quantity_MT': 'sum',
                        'Price_USD_per_kg': 'mean'
                    }).reset_index()
                    monthly_data['Date'] = monthly_data['Date'].dt.to_timestamp()
                    
                    # Remove any NaN values from aggregation
                    monthly_data = monthly_data.dropna()
                    
                    if len(monthly_data) < 10:
                        st.error("Insufficient monthly data for prediction. Need at least 10 months of data.")
                        st.stop()
                    
                    # Prepare features
                    monthly_data['Year'] = monthly_data['Date'].dt.year
                    monthly_data['Month'] = monthly_data['Date'].dt.month
                    monthly_data['Quarter'] = monthly_data['Date'].dt.quarter
                    monthly_data['trend'] = range(len(monthly_data))
                    
                    # Split features for price and volume models
                    X = monthly_data[['Year', 'Month', 'Quarter', 'trend']].copy()
                    y_price = monthly_data['Price_USD_per_kg'].copy()
                    y_volume = monthly_data['Quantity_MT'].copy()
                    
                    # Final check for NaN/inf values
                    if X.isnull().any().any() or y_price.isnull().any() or y_volume.isnull().any():
                        st.error("Data contains missing values after preprocessing. Please check your dataset.")
                        st.stop()
                    
                    if np.isinf(X.values).any() or np.isinf(y_price.values).any() or np.isinf(y_volume.values).any():
                        st.error("Data contains infinite values. Please check your dataset.")
                        st.stop()
                    
                    # Train test split
                    train_size = int(len(monthly_data) * 0.8)
                    if train_size < 5 or (len(monthly_data) - train_size) < 2:
                        st.error("Insufficient data for train-test split. Need at least 10 months of data.")
                        st.stop()
                    
                    X_train = X[:train_size]
                    X_test = X[train_size:]
                    y_price_train = y_price[:train_size]
                    y_price_test = y_price[train_size:]
                    y_volume_train = y_volume[:train_size]
                    y_volume_test = y_volume[train_size:]
                    
                    # Train models
                    from sklearn.ensemble import RandomForestRegressor
                    
                    with st.spinner("Training prediction models..."):
                        # Price model
                        price_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                        price_model.fit(X_train, y_price_train)
                        
                        # Volume model
                        volume_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                        volume_model.fit(X_train, y_volume_train)
                    
                    # Make predictions
                    price_pred = price_model.predict(X_test)
                    volume_pred = volume_model.predict(X_test)
                    
                    # Calculate metrics
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    
                    price_mae = mean_absolute_error(y_price_test, price_pred)
                    price_mse = mean_squared_error(y_price_test, price_pred)
                    price_r2 = r2_score(y_price_test, price_pred)
                    
                    volume_mae = mean_absolute_error(y_volume_test, volume_pred)
                    volume_mse = mean_squared_error(y_volume_test, volume_pred)
                    volume_r2 = r2_score(y_volume_test, volume_pred)
                    
                    # Display results
                    st.markdown("#### 📈 Model Performance")
                    
                    st.info(f"📊 Using {len(monthly_data)} months of data ({train_size} for training, {len(monthly_data)-train_size} for testing)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Price Prediction (USD/kg)")
                        st.metric("Mean Absolute Error", f"${price_mae:.2f}")
                        st.metric("Root Mean Squared Error", f"${np.sqrt(price_mse):.2f}")
                        st.metric("R² Score", f"{price_r2:.2%}")
                        
                    with col2:
                        st.markdown("##### Volume Prediction (MT)")
                        st.metric("Mean Absolute Error", f"{volume_mae:.2f} MT")
                        st.metric("Root Mean Squared Error", f"{np.sqrt(volume_mse):.2f} MT")
                        st.metric("R² Score", f"{volume_r2:.2%}")
                    
                    # Visualizations
                    st.markdown("#### 📊 Predictions vs Actual")
                    
                    # Price predictions plot
                    fig_price = go.Figure()
                    
                    fig_price.add_trace(go.Scatter(
                        x=monthly_data['Date'][train_size:],
                        y=y_price_test,
                        name='Actual Price',
                        line=dict(color='#3b82f6', width=2),
                        mode='lines+markers'
                    ))
                    
                    fig_price.add_trace(go.Scatter(
                        x=monthly_data['Date'][train_size:],
                        y=price_pred,
                        name='Predicted Price',
                        line=dict(color='#ef4444', dash='dash', width=2),
                        mode='lines+markers'
                    ))
                    
                    fig_price.update_layout(
                        title='Price Predictions vs Actual',
                        xaxis_title='Date',
                        yaxis_title='Price (USD/kg)',
                        height=400,
                        showlegend=True,
                        plot_bgcolor='white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_price, use_container_width=True)
                    
                    # Volume predictions plot
                    fig_volume = go.Figure()
                    
                    fig_volume.add_trace(go.Scatter(
                        x=monthly_data['Date'][train_size:],
                        y=y_volume_test,
                        name='Actual Volume',
                        line=dict(color='#10b981', width=2),
                        mode='lines+markers'
                    ))
                    
                    fig_volume.add_trace(go.Scatter(
                        x=monthly_data['Date'][train_size:],
                        y=volume_pred,
                        name='Predicted Volume',
                        line=dict(color='#f97316', dash='dash', width=2),
                        mode='lines+markers'
                    ))
                    
                    fig_volume.update_layout(
                        title='Volume Predictions vs Actual',
                        xaxis_title='Date',
                        yaxis_title='Volume (MT)',
                        height=400,
                        showlegend=True,
                        plot_bgcolor='white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_volume, use_container_width=True)
                    
                    # Future predictions
                    st.markdown("#### 🔮 Future Predictions")
                    
                    num_months = st.slider("Number of months to predict", 1, 12, 3)
                    
                    # Generate future dates
                    last_date = monthly_data['Date'].max()
                    future_dates = pd.date_range(start=last_date, periods=num_months+1, freq='M')[1:]
                    
                    # Create future features
                    future_X = pd.DataFrame({
                        'Year': future_dates.year,
                        'Month': future_dates.month,
                        'Quarter': (future_dates.month - 1) // 3 + 1,
                        'trend': range(len(monthly_data), len(monthly_data) + num_months)
                    })
                    
                    # Make future predictions
                    future_price = price_model.predict(future_X)
                    future_volume = volume_model.predict(future_X)
                    
                    # Ensure predictions are positive
                    future_price = np.maximum(future_price, 0)
                    future_volume = np.maximum(future_volume, 0)
                    
                    # Display predictions table
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price (USD/kg)': future_price,
                        'Predicted Volume (MT)': future_volume
                    })
                    
                    st.dataframe(future_df.style.format({
                        'Predicted Price (USD/kg)': '${:.2f}',
                        'Predicted Volume (MT)': '{:,.2f}'
                    }), use_container_width=True)
                    
                    # Download predictions
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        future_df.to_excel(writer, sheet_name='Future Predictions', index=False)
                        
                        # Add historical predictions
                        historical_pred_df = pd.DataFrame({
                            'Date': monthly_data['Date'][train_size:],
                            'Actual Price (USD/kg)': y_price_test.values,
                            'Predicted Price (USD/kg)': price_pred,
                            'Actual Volume (MT)': y_volume_test.values,
                            'Predicted Volume (MT)': volume_pred
                        })
                        historical_pred_df.to_excel(writer, sheet_name='Historical Predictions', index=False)
                        
                        # Add model metrics
                        metrics_df = pd.DataFrame({
                            'Metric': ['Mean Absolute Error', 'Root Mean Squared Error', 'R² Score'],
                            'Price Model': [price_mae, np.sqrt(price_mse), price_r2],
                            'Volume Model': [volume_mae, np.sqrt(volume_mse), volume_r2]
                        })
                        metrics_df.to_excel(writer, sheet_name='Model Metrics', index=False)
                    
                    output.seek(0)
                    
                    st.download_button(
                        label="📥 Download Predictions",
                        data=output,
                        file_name=f"predictions_{selected_dataset}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                else:
                    st.error("❌ Required columns (date, quantity, value) not found in the dataset")
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                with st.expander("View detailed error"):
                    st.exception(e)

                st.info("💡 Tips: Ensure your data has valid dates, quantities > 0, and values > 0")






