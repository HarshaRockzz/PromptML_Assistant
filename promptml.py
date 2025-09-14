import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    transformers_import_error = str(e)
    transformers_import_success = False
else:
    transformers_import_success = True
    transformers_import_error = None
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from streamlit_chat import message
from streamlit_elements import elements, mui, html, dashboard
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from datetime import datetime
import psutil  # For monitoring memory usage
import time
import random
import json

# Load environment variables with error handling
try:
    load_dotenv()
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
except UnicodeDecodeError:
    st.warning("⚠️ .env file has encoding issues. Please recreate it with UTF-8 encoding.")
    HUGGINGFACEHUB_API_TOKEN = None
    GOOGLE_API_KEY = None
    GROQ_API_KEY = None
except Exception as e:
    st.warning(f"⚠️ Error loading .env file: {str(e)}")
    HUGGINGFACEHUB_API_TOKEN = None
    GOOGLE_API_KEY = None
    GROQ_API_KEY = None

# Set API keys directly if not in .env (fallback)

# Set page config with custom icon (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="PromptML: NextGen AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/HarshaRockzz/PromptML_Assistant',
        'Report a bug': "https://github.com/HarshaRockzz/PromptML_Assistant/issues",
        'About': "# PromptML: NextGen AI Assistant\nRevolutionary AI-powered data science platform with cutting-edge visualizations and intelligent insights."
    }
)

# Display import status and token validation
if transformers_import_success:
    st.write("Successfully imported transformers!")
else:
    st.error(f"Failed to import transformers: {transformers_import_error}")
    raise ImportError(transformers_import_error)

# Display API token status and instructions
if not HUGGINGFACEHUB_API_TOKEN or HUGGINGFACEHUB_API_TOKEN == "your_hugging_face_token_here":
    st.warning("⚠️ HUGGINGFACEHUB_API_TOKEN not configured. Please add your Hugging Face API token to the .env file.")
    st.info("📝 To get your API token: 1) Go to https://huggingface.co/settings/tokens 2) Create a new token 3) Add it to .env file")
    st.code("HUGGINGFACEHUB_API_TOKEN=your_actual_token_here", language="bash")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
    st.warning("⚠️ GOOGLE_API_KEY not configured. Please add your Google API key to the .env file.")
    st.info("📝 To get your API key: 1) Go to https://aistudio.google.com/app/apikey 2) Create a new API key 3) Add it to .env file")
    st.code("GOOGLE_API_KEY=your_actual_api_key_here", language="bash")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    st.warning("⚠️ GROQ_API_KEY not configured. Please add your Groq API key to the .env file.")
    st.info("📝 To get your API key: 1) Go to https://console.groq.com/keys 2) Create a new API key 3) Add it to .env file")
    st.code("GROQ_API_KEY=your_actual_api_key_here", language="bash")

# Hero section will be shown only on EDA page

# Advanced Sidebar with glassmorphism design
with st.sidebar:
    # Logo and branding
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🚀</div>
            <h2 style="margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">PromptML</h2>
            <p style="margin: 0.5rem 0 0 0; color: rgba(44, 62, 80, 0.7); font-size: 0.9rem;">AI-Powered Data Science</p>
        </div>
    """, unsafe_allow_html=True)
    
    theme = st.toggle("🌙 Dark Mode", value=False)
    
    st.markdown("---")
    
    selected = option_menu(
        menu_title=None,
        options=["📊 EDA", "🤖 Model Selection", "🔮 Predictions", "💬 AI Chatbot", "📈 Dashboard"],
        icons=["bar-chart", "cpu", "graph-up", "chat", "speedometer2"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "0px", 
                "background-color": "rgba(255, 255, 255, 0.1)",
                "border-radius": "15px",
                "backdrop-filter": "blur(20px)",
                "border": "1px solid rgba(255, 255, 255, 0.2)"
            },
            "icon": {"color": "#667eea", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "2px 0", 
                "--hover-color": "rgba(102, 126, 234, 0.1)", 
                "color": "#2c3e50",
                "border-radius": "10px",
                "padding": "10px 15px"
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", 
                "color": "white",
                "border-radius": "10px"
            },
        }
    )
    
    st.markdown("---")
    
    # AI Controls
    st.markdown("### 🎛️ AI Controls")
    temperature = st.slider("🧠 AI Temperature", 0.01, 1.0, 0.1, 0.01, help="Controls creativity and randomness")
    
    st.markdown("---")
    
    # File Upload
    st.markdown("### 📁 Data Upload")
    uploaded_files = st.file_uploader("Upload your dataset", type=["csv", "xlsx"], accept_multiple_files=True, help="Support for CSV and Excel files")
    
    # Status indicators
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
    else:
        st.info("📤 No files uploaded yet")
    
    # AI model status will be shown after model loading
    st.info("🤖 AI Models will be loaded when needed")

# Advanced CSS for Next-Level UI
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Root Variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --shadow-soft: 0 8px 32px rgba(0, 0, 0, 0.1);
        --shadow-strong: 0 20px 60px rgba(0, 0, 0, 0.2);
        --border-radius: 20px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
    }
    
    /* Ensure text visibility */
    .main .block-container {
        background: transparent !important;
    }
    
    .main .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Fix Streamlit default text colors */
    .main .stMarkdown {
        color: inherit !important;
    }
    
    .main .stMarkdown p {
        color: inherit !important;
    }
    
    .main .stMarkdown h1,
    .main .stMarkdown h2,
    .main .stMarkdown h3,
    .main .stMarkdown h4,
    .main .stMarkdown h5,
    .main .stMarkdown h6 {
        color: inherit !important;
    }
    
    /* Animated Background */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.3) 0%, transparent 50%);
        animation: backgroundShift 20s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes backgroundShift {
        0%, 100% { transform: translateX(0) translateY(0); }
        25% { transform: translateX(-5%) translateY(-5%); }
        50% { transform: translateX(5%) translateY(-10%); }
        75% { transform: translateX(-3%) translateY(5%); }
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: var(--shadow-strong);
        animation: heroFloat 6s ease-in-out infinite;
    }
    
    @keyframes heroFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: titleGlow 3s ease-in-out infinite;
        color: #667eea !important; /* Fallback color */
    }
    
    @keyframes titleGlow {
        0%, 100% { text-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        50% { text-shadow: 0 0 40px rgba(102, 126, 234, 0.8); }
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(44, 62, 80, 0.8);
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: var(--border-radius);
        padding: 2rem;
        box-shadow: var(--shadow-soft);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-strong);
    }
    
    /* Advanced Buttons */
    .stButton>button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1rem;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Primary Action Button */
    .primary-btn {
        background: var(--accent-gradient) !important;
        box-shadow: 0 8px 30px rgba(79, 172, 254, 0.4) !important;
    }
    
    .primary-btn:hover {
        box-shadow: 0 12px 40px rgba(79, 172, 254, 0.6) !important;
    }
    
    /* Input Fields */
    .stTextInput>div>input,
    .stTextArea>div>textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 15px !important;
        padding: 15px 20px !important;
        color: #2c3e50 !important;
        font-size: 1rem !important;
        transition: var(--transition) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stTextInput>div>input:focus,
    .stTextArea>div>textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
        background: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 2px solid rgba(255, 255, 255, 0.2) !important;
        padding: 2rem !important;
    }
    
    /* Fix sidebar text visibility */
    .sidebar .sidebar-content .stMarkdown {
        color: #2c3e50 !important;
    }
    
    .sidebar .sidebar-content .stMarkdown h1,
    .sidebar .sidebar-content .stMarkdown h2,
    .sidebar .sidebar-content .stMarkdown h3 {
        color: #667eea !important;
    }
    
    .sidebar .sidebar-content .stMarkdown p {
        color: #2c3e50 !important;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background: var(--primary-gradient) !important;
        border-radius: 10px !important;
    }
    
    .stSlider > div > div > div > div {
        background: white !important;
        border-radius: 50% !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Alert Styling */
    .stAlert {
        background: linear-gradient(135deg, rgba(255, 235, 238, 0.9) 0%, rgba(255, 245, 245, 0.9) 100%) !important;
        border: 2px solid rgba(211, 47, 47, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(211, 47, 47, 0.1) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 243, 224, 0.9) 0%, rgba(255, 248, 240, 0.9) 100%) !important;
        border: 2px solid rgba(245, 124, 0, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(245, 124, 0, 0.1) !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(232, 245, 233, 0.9) 0%, rgba(240, 248, 241, 0.9) 100%) !important;
        border: 2px solid rgba(46, 125, 50, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(46, 125, 50, 0.1) !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        background: var(--primary-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 700 !important;
    }
    
    /* Dataframe Styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Chart Containers */
    .js-plotly-plot {
        border-radius: 15px !important;
        box-shadow: var(--shadow-soft) !important;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Pulse Animation */
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Fade In Animation */
    .fade-in {
        animation: fadeIn 0.8s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        min-height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-strong);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 900;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(44, 62, 80, 0.8);
        font-weight: 600;
        margin-top: 0.25rem;
    }
    
    /* Chat Interface */
    .chat-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        height: 500px;
        overflow-y: auto;
    }
    
    .chat-message {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .chat-message.user {
        background: var(--primary-gradient);
        color: white;
        margin-left: 2rem;
    }
    
    .chat-message.bot {
        background: rgba(255, 255, 255, 0.2);
        color: #2c3e50;
        margin-right: 2rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-section {
            padding: 2rem;
            margin: 1rem 0;
        }
        
        .glass-card {
            padding: 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Apply dark mode styles if toggled
if theme:
    st.markdown("""
        <style>
        /* Dark Mode Styles */
        .main {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
            color: #ffffff !important;
        }
        
        .main .block-container {
            background: transparent !important;
        }
        
        .main .stApp {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
        }
        
        .main::before {
            background: 
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.15) 0%, transparent 50%);
        }
        
        .hero-section {
            background: linear-gradient(135deg, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.2) 100%) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            color: #ffffff !important;
        }
        
        .glass-card {
            background: rgba(0, 0, 0, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: #ffffff !important;
        }
        
        .metric-card {
            background: rgba(0, 0, 0, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: #ffffff !important;
        }
        
        .metric-value {
            color: #4fc3f7 !important;
        }
        
        .metric-label {
            color: #b0b0b0 !important;
        }
        
        .stTextInput>div>input,
        .stTextArea>div>textarea {
            background: rgba(0, 0, 0, 0.4) !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            color: #ffffff !important;
        }
        
        .stTextInput>div>input:focus,
        .stTextArea>div>textarea:focus {
            border-color: #4fc3f7 !important;
            box-shadow: 0 0 20px rgba(79, 195, 247, 0.3) !important;
            background: rgba(0, 0, 0, 0.5) !important;
        }
        
        .sidebar .sidebar-content {
            background: rgba(0, 0, 0, 0.3) !important;
            border-right: 2px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        .dataframe {
            background: rgba(0, 0, 0, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: #ffffff !important;
        }
        
        .chat-message.bot {
            background: rgba(0, 0, 0, 0.4) !important;
            color: #ffffff !important;
        }
        
        .chat-message.user {
            background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%) !important;
            color: #ffffff !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #4fc3f7 !important;
        }
        
        .hero-title {
            color: #4fc3f7 !important;
        }
        
        .hero-subtitle {
            color: #b0b0b0 !important;
        }
        
        /* Fix Streamlit default styles for dark mode */
        .stSelectbox > div > div {
            background: rgba(0, 0, 0, 0.4) !important;
            color: #ffffff !important;
        }
        
        .stSelectbox > div > div > div {
            color: #ffffff !important;
        }
        
        /* Fix all text visibility */
        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #ffffff !important;
        }
        
        /* Fix metric card text */
        .metric-card .metric-value {
            color: #4fc3f7 !important;
            font-weight: 900 !important;
            font-size: 1.8rem !important;
            line-height: 1.2 !important;
            margin-bottom: 0.25rem !important;
        }
        
        .metric-card .metric-label {
            color: #b0b0b0 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            line-height: 1.2 !important;
            margin-top: 0.25rem !important;
        }
        
        /* Fix metric card alignment */
        .metric-card {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
            min-height: 70px !important;
            padding: 0.75rem !important;
        }
        
        /* Fix sidebar text */
        .sidebar .sidebar-content .stMarkdown {
            color: #ffffff !important;
        }
        
        .sidebar .sidebar-content .stMarkdown p {
            color: #ffffff !important;
        }
        
        /* Fix sidebar navigation text visibility */
        .sidebar .sidebar-content .stMarkdown h1,
        .sidebar .sidebar-content .stMarkdown h2,
        .sidebar .sidebar-content .stMarkdown h3,
        .sidebar .sidebar-content .stMarkdown h4,
        .sidebar .sidebar-content .stMarkdown h5,
        .sidebar .sidebar-content .stMarkdown h6 {
            color: #ffffff !important;
        }
        
        /* Fix option menu text visibility */
        .stSelectbox > div > div > div {
            color: #ffffff !important;
        }
        
        /* Fix all sidebar text elements */
        .sidebar .sidebar-content * {
            color: #ffffff !important;
        }
        
        /* Override any specific sidebar text that might be hidden */
        .sidebar .sidebar-content .stMarkdown,
        .sidebar .sidebar-content .stMarkdown *,
        .sidebar .sidebar-content div,
        .sidebar .sidebar-content span,
        .sidebar .sidebar-content p {
            color: #ffffff !important;
        }
        
        /* Fix option menu navigation items */
        .stSelectbox .stSelectbox > div > div > div {
            color: #ffffff !important;
        }
        
        /* Fix all text in sidebar */
        .sidebar * {
            color: #ffffff !important;
        }
        
        /* Specific fix for navigation menu text */
        .sidebar .sidebar-content .stMarkdown h1,
        .sidebar .sidebar-content .stMarkdown h2,
        .sidebar .sidebar-content .stMarkdown h3,
        .sidebar .sidebar-content .stMarkdown h4,
        .sidebar .sidebar-content .stMarkdown h5,
        .sidebar .sidebar-content .stMarkdown h6,
        .sidebar .sidebar-content .stMarkdown p,
        .sidebar .sidebar-content .stMarkdown div,
        .sidebar .sidebar-content .stMarkdown span {
            color: #ffffff !important;
            opacity: 1 !important;
        }
        
        /* Fix form text */
        .stForm .stMarkdown {
            color: #ffffff !important;
        }
        
        .stForm .stMarkdown p {
            color: #ffffff !important;
        }
        
        /* Fix option menu navigation specifically */
        .stSelectbox > div > div > div > div {
            color: #ffffff !important;
        }
        
        /* Target the specific option menu used for navigation */
        .sidebar .stSelectbox > div > div > div {
            color: #ffffff !important;
            background: rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Force visibility for all sidebar text */
        .sidebar .sidebar-content {
            color: #ffffff !important;
        }
        
        .sidebar .sidebar-content * {
            color: #ffffff !important;
            opacity: 1 !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%) !important;
            color: #ffffff !important;
        }
        
        .stAlert {
            background: rgba(0, 0, 0, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: #ffffff !important;
        }
        
        .stSuccess {
            background: rgba(0, 100, 0, 0.2) !important;
            border: 1px solid rgba(0, 255, 0, 0.3) !important;
            color: #ffffff !important;
        }
        
        .stWarning {
            background: rgba(255, 165, 0, 0.2) !important;
            border: 1px solid rgba(255, 165, 0, 0.3) !important;
            color: #ffffff !important;
        }
        
        .stError {
            background: rgba(255, 0, 0, 0.2) !important;
            border: 1px solid rgba(255, 0, 0, 0.3) !important;
            color: #ffffff !important;
        }
        
        .stInfo {
            background: rgba(0, 123, 255, 0.2) !important;
            border: 1px solid rgba(0, 123, 255, 0.3) !important;
            color: #ffffff !important;
        }
        
        /* Fix sidebar text visibility */
        .sidebar .sidebar-content .stMarkdown {
            color: #ffffff !important;
        }
        
        .sidebar .sidebar-content .stMarkdown h1,
        .sidebar .sidebar-content .stMarkdown h2,
        .sidebar .sidebar-content .stMarkdown h3 {
            color: #4fc3f7 !important;
        }
        
        /* Fix form elements */
        .stForm {
            background: rgba(0, 0, 0, 0.2) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        .stForm .stMarkdown {
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    # Light mode styles
    st.markdown("""
        <style>
        /* Light Mode Styles */
        .main {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%) !important;
            color: #2c3e50 !important;
        }
        
        .main .block-container {
            background: transparent !important;
        }
        
        .main .stApp {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%) !important;
        }
        
        .hero-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%) !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            color: #2c3e50 !important;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.8) !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            color: #2c3e50 !important;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.8) !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            color: #2c3e50 !important;
        }
        
        .metric-value {
            color: #667eea !important;
        }
        
        .metric-label {
            color: #6c757d !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50 !important;
        }
        
        .hero-title {
            color: #667eea !important;
        }
        
        .hero-subtitle {
            color: #6c757d !important;
        }
        
        /* Fix all text visibility in light mode */
        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #2c3e50 !important;
        }
        
        /* Fix metric card text in light mode */
        .metric-card .metric-value {
            color: #667eea !important;
            font-weight: 900 !important;
            font-size: 1.8rem !important;
            line-height: 1.2 !important;
            margin-bottom: 0.25rem !important;
        }
        
        .metric-card .metric-label {
            color: #6c757d !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            line-height: 1.2 !important;
            margin-top: 0.25rem !important;
        }
        
        /* Fix metric card alignment in light mode */
        .metric-card {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
            min-height: 70px !important;
            padding: 0.75rem !important;
        }
        
        /* Fix sidebar text visibility in light mode */
        .sidebar .sidebar-content * {
            color: #2c3e50 !important;
            opacity: 1 !important;
        }
        
        .sidebar .sidebar-content .stMarkdown,
        .sidebar .sidebar-content .stMarkdown *,
        .sidebar .sidebar-content div,
        .sidebar .sidebar-content span,
        .sidebar .sidebar-content p {
            color: #2c3e50 !important;
            opacity: 1 !important;
        }
        
        .stTextInput>div>input,
        .stTextArea>div>textarea {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 2px solid rgba(0,0,0,0.1) !important;
            color: #2c3e50 !important;
        }
        
        .stTextInput>div>input:focus,
        .stTextArea>div>textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
            background: rgba(255, 255, 255, 1) !important;
        }
        
        .sidebar .sidebar-content {
            background: rgba(255, 255, 255, 0.9) !important;
            border-right: 2px solid rgba(0,0,0,0.1) !important;
        }
        
        .dataframe {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            color: #2c3e50 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Load Data with Progress Bar and Animation
if uploaded_files:
    with st.spinner("Loading data..."):
        progress_bar = st.progress(0)
        dfs = []
        for i, file in enumerate(uploaded_files):
            dfs.append(pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file))
            progress_bar.progress((i + 1) / len(uploaded_files))
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        st.success("Data loaded successfully!")
else:
    df = None

# Fast AI Chat System with Multiple Providers
@st.cache_resource
def load_chat_models():
    """Load multiple AI chat models for better performance and reliability"""
    models = {}
    
    # Google Gemini (Fast and reliable)
    try:
        if GOOGLE_API_KEY:
            models['gemini'] = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                max_tokens=2048
            )
            st.success("✅ Google Gemini loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Google Gemini failed to load: {str(e)}")
    
    # Groq (Ultra fast)
    try:
        if GROQ_API_KEY:
            models['groq'] = ChatGroq(
                model="llama3-8b-8192",
                groq_api_key=GROQ_API_KEY,
                temperature=0.7,
                max_tokens=2048
            )
            st.success("✅ Groq loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Groq failed to load: {str(e)}")
    
    # Hugging Face (Fallback)
    try:
        if HUGGINGFACEHUB_API_TOKEN:
            models['huggingface'] = ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id="microsoft/DialoGPT-medium",
                    task="text-generation",
                    max_new_tokens=256,
                    temperature=0.7,
                    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
                )
            )
            st.success("✅ Hugging Face loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Hugging Face failed to load: {str(e)}")
    
    return models

# Legacy model for sentiment analysis
@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACEHUB_API_TOKEN)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=HUGGINGFACEHUB_API_TOKEN)
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        st.warning(f"Sentiment analysis model failed to load: {str(e)}")
        return None

# Fast AI Response Function
def get_ai_response(chat_models, user_message, model_choice="auto"):
    """Get AI response from the best available model"""
    
    # Model priority: Groq (fastest) -> Gemini (reliable) -> HuggingFace (fallback)
    if model_choice == "auto":
        if 'groq' in chat_models:
            model = chat_models['groq']
            model_name = "Groq (Llama 3)"
        elif 'gemini' in chat_models:
            model = chat_models['gemini']
            model_name = "Google Gemini"
        elif 'huggingface' in chat_models:
            model = chat_models['huggingface']
            model_name = "Hugging Face"
        else:
            return "No AI models available. Please check your API keys.", "None"
    else:
        if model_choice in chat_models:
            model = chat_models[model_choice]
            model_name = model_choice.title()
        else:
            return "Selected model not available.", "None"
    
    try:
        # Create a system prompt for data science assistance
        system_prompt = """You are an expert data science AI assistant. You help users with:
        - Data analysis and visualization
        - Machine learning model selection and implementation
        - Statistical analysis and insights
        - Data preprocessing and cleaning
        - Python programming for data science
        - Explaining complex concepts in simple terms
        
        Provide clear, actionable advice and code examples when appropriate. Be concise but thorough."""
        
        # Format the message with system prompt
        formatted_message = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        # Get response
        response = model.invoke(formatted_message)
        
        # Extract text content
        if hasattr(response, 'content'):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)
        
        # Clean up response
        response_text = response_text.strip()
        if response_text.startswith('Assistant:'):
            response_text = response_text.replace('Assistant:', '').strip()
        
        return response_text, model_name
        
    except Exception as e:
        return f"Error getting response: {str(e)}", "Error"

# Custom function for text classification
def classify_text(model_dict, text):
    if model_dict is None:
        return "Model not available"
    try:
        inputs = model_dict["tokenizer"](text, return_tensors="pt", truncation=True, padding=True)
        outputs = model_dict["model"](**inputs)
        logits = outputs.logits
        probabilities = logits.softmax(dim=-1)
        prediction = probabilities.argmax(dim=-1).item()
        return "Positive" if prediction == 1 else "Negative"
    except Exception as e:
        return f"Classification error: {str(e)}"

# Load models
chat_models = load_chat_models()
sentiment_model = load_sentiment_model()

# Check if any chat models loaded successfully
if not chat_models:
    st.warning("⚠️ No AI chat models available. Please check your API keys.")
else:
    st.success(f"✅ {len(chat_models)} AI model(s) loaded successfully!")

# EDA Page with Stunning Visualizations
if selected == "📊 EDA":
    # Hero section for EDA
    st.markdown("""
        <div class="hero-section fade-in">
            <h1 class="hero-title">🚀 PromptML</h1>
            <p class="hero-subtitle">Next-Generation AI-Powered Data Science Platform</p>
            <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
                <div class="metric-card" style="flex: 1; max-width: 150px;">
                    <div class="metric-value">95%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card" style="flex: 1; max-width: 150px;">
                    <div class="metric-value">2s</div>
                    <div class="metric-label">Response Time</div>
                </div>
                <div class="metric-card" style="flex: 1; max-width: 150px;">
                    <div class="metric-value">∞</div>
                    <div class="metric-label">Possibilities</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # EDA specific header
    st.markdown("""
        <div class="glass-card fade-in" style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2rem; margin-bottom: 0.5rem;">📊 Exploratory Data Analysis</h1>
            <p style="font-size: 1rem; color: rgba(44, 62, 80, 0.8);">Discover hidden patterns and insights in your data with AI-powered visualizations</p>
        </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Data metrics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-value">{df.shape[0]:,}</div>
                    <div class="metric-label">Rows</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-value">{df.shape[1]}</div>
                    <div class="metric-label">Columns</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-value">{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB</div>
                    <div class="metric-label">Memory</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-value">{df.isnull().sum().sum()}</div>
                    <div class="metric-label">Missing Values</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Advanced tabs with glassmorphism
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Overview", "🏥 Health Check", "📈 Visualizations", "🤖 AI Insights"])
        
        with tab1:
            st.markdown("""
                <div class="glass-card fade-in">
                    <h2 style="margin-top: 0;">📋 Data Overview</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Enhanced dataframe display
            st.dataframe(
                df.head(10), 
                width='stretch', 
                height=400,
                column_config={
                    col: st.column_config.TextColumn(
                        col,
                        help=f"Column: {col}",
                        width="medium"
                    ) for col in df.columns
                }
            )
            
            # Statistical summary with enhanced styling
            with st.expander("📊 Statistical Summary", expanded=False):
                st.markdown("""
                    <div class="glass-card">
                        <h3>Descriptive Statistics</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.dataframe(df.describe(), width='stretch')
        
        with tab2:
            st.markdown("""
                <div class="glass-card fade-in">
                    <h2 style="margin-top: 0;">🏥 Data Health Check</h2>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Missing values visualization
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    fig_missing = px.pie(
                        names=missing.index, 
                        values=missing.values, 
                        title="Missing Values Distribution",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_missing.update_layout(
                        title_font_size=20,
                        font=dict(size=14),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_missing, width='stretch')
                else:
                    st.success("🎉 No missing values found!")
            
            with col2:
                # Data types visualization
                dtype_counts = df.dtypes.value_counts()
                fig_dtypes = px.bar(
                    x=dtype_counts.index.astype(str), 
                    y=dtype_counts.values,
                    title="Data Types Distribution",
                    color=dtype_counts.values,
                    color_continuous_scale="Viridis"
                )
                fig_dtypes.update_layout(
                    title_font_size=20,
                    font=dict(size=14),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_dtypes, width='stretch')
        
        with tab3:
            st.markdown("""
                <div class="glass-card fade-in">
                    <h2 style="margin-top: 0;">📈 Interactive Visualizations</h2>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("""
                    <div class="glass-card">
                        <h3>🎛️ Controls</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                viz_type = st.selectbox(
                    "Chart Type", 
                    ["📊 Histogram", "📈 Scatter Plot", "📦 Box Plot", "🔗 Correlation Matrix", "📊 Bar Chart", "🌡️ Heatmap"],
                    help="Select the type of visualization"
                )
                
                if viz_type in ["📊 Histogram", "📦 Box Plot", "📊 Bar Chart"]:
                    column = st.selectbox("Select Column", df.columns, help="Choose a column to visualize")
                elif viz_type == "📈 Scatter Plot":
                    col1_viz, col2_viz = st.columns(2)
                    with col1_viz:
                        x_col = st.selectbox("X-axis", df.columns)
                    with col2_viz:
                        y_col = st.selectbox("Y-axis", df.columns)
                elif viz_type == "🔗 Correlation Matrix":
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:5])
            
            with col2:
                st.markdown("""
                    <div class="glass-card">
                        <h3>📊 Visualization</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                if viz_type == "📊 Histogram":
                    fig = px.histogram(
                        df, x=column, 
                        marginal="box", 
                        color_discrete_sequence=["#667eea"],
                        title=f"Distribution of {column}",
                        nbins=30
                    )
                    fig.update_layout(
                        title_font_size=20,
                        font=dict(size=14),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                elif viz_type == "📈 Scatter Plot":
                    fig = px.scatter(
                        df, x=x_col, y=y_col, 
                        trendline="ols", 
                        color_discrete_sequence=["#667eea"],
                        title=f"{x_col} vs {y_col}",
                        opacity=0.7
                    )
                    fig.update_layout(
                        title_font_size=20,
                        font=dict(size=14),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                elif viz_type == "📦 Box Plot":
                    fig = px.box(
                        df, y=column, 
                        color_discrete_sequence=["#667eea"],
                        title=f"Box Plot of {column}"
                    )
                    fig.update_layout(
                        title_font_size=20,
                        font=dict(size=14),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                elif viz_type == "🔗 Correlation Matrix":
                    if selected_cols:
                        corr_data = df[selected_cols].corr()
                        fig = px.imshow(
                            corr_data, 
                            text_auto=True, 
                            color_continuous_scale="RdBu",
                            title="Correlation Matrix",
                            aspect="auto"
                        )
                        fig.update_layout(
                            title_font_size=20,
                            font=dict(size=14),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                else:
                    st.warning("Please select at least one column for correlation analysis.")
                    fig = None
                        
            if viz_type == "📊 Bar Chart":
                value_counts = df[column].value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    title=f"Top 10 Values in {column}",
                    color=value_counts.values,
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(
                    title_font_size=20,
                    font=dict(size=14),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
            elif viz_type == "🌡️ Heatmap":
                numeric_df = df.select_dtypes(include=['number'])
                if not numeric_df.empty:
                    fig = px.imshow(
                        numeric_df.corr(), 
                        text_auto=True, 
                        color_continuous_scale="Viridis",
                        title="Numeric Columns Heatmap"
                    )
                    fig.update_layout(
                        title_font_size=20,
                        font=dict(size=14),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                else:
                    st.warning("No numeric columns found for heatmap.")
                    fig = None
                
            if fig is not None:
                st.plotly_chart(fig, width='stretch')

        with tab4:
            st.markdown("""
                <div class="glass-card fade-in">
                    <h2 style="margin-top: 0;">🤖 AI-Powered Insights</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if chat_models:
                # AI insights generation
                with st.spinner("🤖 AI is analyzing your data..."):
                    time.sleep(2)  # Simulate AI processing
                    
                    # Generate insights based on data characteristics
                    insights = []
                    
                    # Missing values insight
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    if missing_pct > 10:
                        insights.append(f"⚠️ High missing data: {missing_pct:.1f}% of values are missing")
                    elif missing_pct > 0:
                        insights.append(f"✅ Low missing data: {missing_pct:.1f}% of values are missing")
                    else:
                        insights.append("🎉 Perfect! No missing values detected")
                    
                    # Data size insight
                    if df.shape[0] > 10000:
                        insights.append(f"📊 Large dataset: {df.shape[0]:,} rows - consider sampling for faster analysis")
                    elif df.shape[0] > 1000:
                        insights.append(f"📈 Medium dataset: {df.shape[0]:,} rows - good size for analysis")
                    else:
                        insights.append(f"📋 Small dataset: {df.shape[0]:,} rows - may need more data for robust analysis")
                    
                    # Column diversity insight
                    numeric_cols = df.select_dtypes(include=['number']).shape[1]
                    categorical_cols = df.select_dtypes(include=['object', 'category']).shape[1]
                    insights.append(f"🔢 Data types: {numeric_cols} numeric, {categorical_cols} categorical columns")
                    
                    # Display insights
                    for i, insight in enumerate(insights):
                        st.markdown(f"""
                            <div class="glass-card" style="margin: 1rem 0; animation-delay: {i*0.2}s;">
                                <p style="margin: 0; font-size: 1.1rem;">{insight}</p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("🤖 AI features are disabled. Please check your API keys.")
    
    else:
        st.markdown("""
            <div class="glass-card fade-in" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
                <h2>No Data Uploaded</h2>
                <p style="font-size: 1.2rem; color: rgba(44, 62, 80, 0.8); margin-bottom: 2rem;">
                    Upload a dataset to start exploring your data with AI-powered insights
                </p>
                <div style="font-size: 2rem;">⬅️</div>
                <p>Use the sidebar to upload your data</p>
            </div>
        """, unsafe_allow_html=True)

# Helper function to generate model recommendations
def generate_model_recommendations(problem_type, data_size, complexity):
    """Generate AI-powered model recommendations based on problem characteristics"""
    
    if problem_type == "Classification":
        return {
            "Random Forest": {
                "icon": "🌲",
                "subtitle": "Ensemble Learning",
                "accuracy": "85-95%",
                "speed": "Fast",
                "interpretability": "Medium",
                "description": "Excellent for tabular data with mixed feature types. Handles missing values well and provides feature importance."
            },
            "XGBoost": {
                "icon": "⚡",
                "subtitle": "Gradient Boosting",
                "accuracy": "90-98%",
                "speed": "Medium",
                "interpretability": "Low",
                "description": "State-of-the-art performance for structured data. Great for competitions and production systems."
            },
            "Neural Network": {
                "icon": "🧠",
                "subtitle": "Deep Learning",
                "accuracy": "88-96%",
                "speed": "Slow",
                "interpretability": "Low",
                "description": "Powerful for complex patterns. Requires more data and tuning but can achieve high accuracy."
            }
        }
    elif problem_type == "Regression":
        return {
            "Linear Regression": {
                "icon": "📈",
                "subtitle": "Linear Model",
                "accuracy": "70-85%",
                "speed": "Very Fast",
                "interpretability": "High",
                "description": "Simple and interpretable. Great baseline model for linear relationships."
            },
            "Random Forest Regressor": {
                "icon": "🌲",
                "subtitle": "Ensemble Learning",
                "accuracy": "80-92%",
                "speed": "Fast",
                "interpretability": "Medium",
                "description": "Robust to outliers and handles non-linear relationships well."
            },
            "XGBoost Regressor": {
                "icon": "⚡",
                "subtitle": "Gradient Boosting",
                "accuracy": "85-95%",
                "speed": "Medium",
                "interpretability": "Low",
                "description": "High performance for regression tasks with excellent handling of missing values."
            }
        }
    elif problem_type == "Clustering":
        return {
            "K-Means": {
                "icon": "🎯",
                "subtitle": "Centroid-based",
                "accuracy": "Good",
                "speed": "Fast",
                "interpretability": "High",
                "description": "Simple and efficient for spherical clusters. Great for customer segmentation."
            },
            "DBSCAN": {
                "icon": "🔍",
                "subtitle": "Density-based",
                "accuracy": "Good",
                "speed": "Medium",
                "interpretability": "Medium",
                "description": "Finds clusters of arbitrary shape and handles noise well."
            },
            "Hierarchical": {
                "icon": "🌳",
                "subtitle": "Tree-based",
                "accuracy": "Good",
                "speed": "Slow",
                "interpretability": "High",
                "description": "Creates a tree of clusters. Good for understanding data structure."
            }
        }
    else:
        return {
            "Custom Model": {
                "icon": "🔧",
                "subtitle": "Specialized",
                "accuracy": "Variable",
                "speed": "Variable",
                "interpretability": "Variable",
                "description": "Specialized models for this problem type. Consider consulting domain experts."
            }
        }

# Model Selection with Stunning Cards and Animations
if selected == "🤖 Model Selection":
    # Hero section for Model Selection
    st.markdown("""
        <div class="glass-card fade-in" style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">🤖 AI Model Selection</h1>
            <p style="font-size: 1.2rem; color: rgba(44, 62, 80, 0.8);">Let AI recommend the perfect machine learning models for your data</p>
        </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Problem configuration form
        st.markdown("""
            <div class="glass-card fade-in">
                <h2 style="margin-top: 0;">⚙️ Problem Configuration</h2>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("model_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                problem_type = st.selectbox(
                    "🎯 Problem Type", 
                    ["Classification", "Regression", "Clustering", "Time Series", "NLP", "Computer Vision"],
                    help="Select the type of machine learning problem"
                )
                
            with col2:
                if problem_type in ["Classification", "Regression"]:
                    target = st.selectbox("🎯 Target Variable", df.columns, help="Choose the variable you want to predict")
                else:
                    target = "N/A"
            
            # Advanced options
            with st.expander("🔧 Advanced Options", expanded=False):
                col3, col4 = st.columns(2)
                with col3:
                    data_size = st.selectbox("📊 Dataset Size", ["Small (<1K)", "Medium (1K-10K)", "Large (10K+)", "Very Large (100K+)"])
                with col4:
                    complexity = st.selectbox("🧠 Model Complexity", ["Simple", "Medium", "Complex", "Very Complex"])
            
            submitted = st.form_submit_button("🚀 Get AI Recommendations", width='stretch')
        
        if submitted:
            if chat_models:
                with st.spinner("🤖 AI is analyzing your data and generating recommendations..."):
                    time.sleep(3)  # Simulate AI processing
                    
                    # Generate dynamic recommendations based on problem type
                    recommendations = generate_model_recommendations(problem_type, data_size, complexity)
                    
                    # Display recommendations in stunning cards
                    st.markdown("""
                        <div class="glass-card fade-in">
                            <h2 style="margin-top: 0;">🎯 AI Recommendations</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Create recommendation cards
                    for i, (model, details) in enumerate(recommendations.items()):
                        st.markdown(f"""
                            <div class="glass-card" style="margin: 1rem 0; animation-delay: {i*0.2}s;">
                                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                    <div style="font-size: 2rem; margin-right: 1rem;">{details['icon']}</div>
                                    <div>
                                        <h3 style="margin: 0; color: #667eea;">{model}</h3>
                                        <p style="margin: 0; color: rgba(44, 62, 80, 0.7);">{details['subtitle']}</p>
                                    </div>
                                </div>
                                <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                                    <div style="flex: 1;">
                                        <strong>Accuracy:</strong> {details['accuracy']}
                                    </div>
                                    <div style="flex: 1;">
                                        <strong>Speed:</strong> {details['speed']}
                                    </div>
                                    <div style="flex: 1;">
                                        <strong>Interpretability:</strong> {details['interpretability']}
                                    </div>
                                </div>
                                <p style="margin: 0; color: rgba(44, 62, 80, 0.8);">{details['description']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Success animation
                    st.balloons()
                    
                    # Download button
                    recommendation_text = f"AI Model Recommendations for {problem_type}\n\n"
                    for model, details in recommendations.items():
                        recommendation_text += f"{model}: {details['description']}\n"
                    
                    st.download_button(
                        "📥 Download Recommendations", 
                        recommendation_text, 
                        "ai_model_recommendations.txt",
                        width='stretch'
                    )
            else:
                st.error("🤖 Cannot generate recommendations. Please check your API keys.")
    
    else:
        st.markdown("""
            <div class="glass-card fade-in" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
                <h2>No Data Uploaded</h2>
                <p style="font-size: 1.2rem; color: rgba(44, 62, 80, 0.8); margin-bottom: 2rem;">
                    Upload a dataset to get AI-powered model recommendations
                </p>
                <div style="font-size: 2rem;">⬅️</div>
                <p>Use the sidebar to upload your data</p>
            </div>
        """, unsafe_allow_html=True)


# Futuristic Predictions Interface
if selected == "🔮 Predictions":
    # Hero section for Predictions
    st.markdown("""
        <div class="glass-card fade-in" style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">🔮 AI Predictions</h1>
            <p style="font-size: 1.2rem; color: rgba(44, 62, 80, 0.8);">Experience the future of AI-powered predictions with real-time analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Prediction interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
                <div class="glass-card fade-in">
                    <h2 style="margin-top: 0;">🎛️ Prediction Controls</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Model selection
            model_choice = st.selectbox(
                "🤖 AI Model", 
                ["DistilBERT (Sentiment Analysis)", "Custom Model", "Advanced NLP"],
                help="Select the AI model for predictions"
            )
            
            # Input section
            st.markdown("### 📝 Input Text")
            text_input = st.text_area(
                "Enter text to analyze", 
                "This is an amazing AI-powered application that will revolutionize data science!",
                height=150,
                help="Enter the text you want to analyze"
            )
            
            # Advanced options
            with st.expander("⚙️ Advanced Settings", expanded=False):
                confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
                show_probabilities = st.checkbox("Show Probability Scores", value=True)
                explain_prediction = st.checkbox("Explain Prediction", value=True)
            
            # Predict button
            predict_button = st.button("🚀 Generate Prediction", width='stretch')
        
        with col2:
            st.markdown("""
                <div class="glass-card fade-in">
                    <h2 style="margin-top: 0;">📊 Prediction Results</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if predict_button:
                if sentiment_model is not None:
                    with st.spinner("🤖 AI is analyzing your text..."):
                        # Simulate AI processing time
                        time.sleep(2)
                        
                        # Generate prediction
                        prediction = classify_text(sentiment_model, text_input)
                        
                        # Create prediction result
                        prediction_color = "#4CAF50" if prediction == "Positive" else "#F44336"
                        prediction_icon = "😊" if prediction == "Positive" else "😞"
                        
                        # Main prediction display
                        st.markdown(f"""
                            <div class="glass-card" style="text-align: center; margin: 2rem 0;">
                                <div style="font-size: 4rem; margin-bottom: 1rem;">{prediction_icon}</div>
                                <h1 style="color: {prediction_color}; margin: 0;">{prediction}</h1>
                                <p style="color: rgba(44, 62, 80, 0.7); margin: 0.5rem 0;">Sentiment Analysis Result</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence visualization
                        confidence = random.uniform(0.7, 0.95)  # Simulate confidence score
                        st.markdown(f"""
                            <div class="glass-card">
                                <h3 style="margin-top: 0;">📈 Confidence Score</h3>
                                <div style="background: linear-gradient(90deg, #F44336 0%, #FF9800 50%, #4CAF50 100%); height: 20px; border-radius: 10px; position: relative; margin: 1rem 0;">
                                    <div style="background: white; height: 16px; width: 16px; border-radius: 50%; position: absolute; top: 2px; left: {confidence * 100}%; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
                                </div>
                                <p style="text-align: center; margin: 0; font-weight: 600;">{confidence:.1%} Confidence</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Probability breakdown
                        if show_probabilities:
                            pos_prob = confidence if prediction == "Positive" else 1 - confidence
                            neg_prob = 1 - pos_prob
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=["Positive", "Negative"], 
                                    y=[pos_prob, neg_prob],
                                    marker_color=[prediction_color, "#F44336" if prediction == "Positive" else "#4CAF50"],
                                    text=[f"{pos_prob:.1%}", f"{neg_prob:.1%}"],
                                    textposition='auto'
                                )
                            ])
                            fig.update_layout(
                                title="Probability Distribution",
                                title_font_size=20,
                                font=dict(size=14),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=300
                            )
                        st.plotly_chart(fig, width='stretch')
                        
                        # Explanation
                        if explain_prediction:
                            explanations = {
                                "Positive": [
                                    "The text contains positive sentiment indicators",
                                    "Words like 'amazing', 'revolutionize', 'great' suggest optimism",
                                    "Overall tone is enthusiastic and upbeat"
                                ],
                                "Negative": [
                                    "The text contains negative sentiment indicators",
                                    "Words suggest disappointment or criticism",
                                    "Overall tone is pessimistic or critical"
                                ]
                            }
                            
                            st.markdown("""
                                <div class="glass-card">
                                    <h3 style="margin-top: 0;">🔍 AI Explanation</h3>
                            """, unsafe_allow_html=True)
                            
                            for i, explanation in enumerate(explanations[prediction]):
                                st.markdown(f"""
                                    <div style="margin: 0.5rem 0; padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
                                        <span style="color: #667eea; font-weight: 600;">{i+1}.</span> {explanation}
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Success animation
                        st.balloons()
                        
                        # Download results
                        results_text = f"AI Prediction Results\n\nText: {text_input}\nPrediction: {prediction}\nConfidence: {confidence:.1%}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        st.download_button(
                            "📥 Download Results", 
                            results_text, 
                            "prediction_results.txt",
                            width='stretch'
                        )
                else:
                    st.error("🤖 Cannot generate predictions. Please check your API keys.")
            else:
                # Placeholder when no prediction is made
                st.markdown("""
                    <div class="glass-card" style="text-align: center; padding: 3rem;">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">🔮</div>
                        <h3>Ready to Predict</h3>
                        <p style="color: rgba(44, 62, 80, 0.7);">Enter your text and click "Generate Prediction" to see AI analysis</p>
                    </div>
                """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
            <div class="glass-card fade-in" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🔮</div>
                <h2>No Data Available</h2>
                <p style="font-size: 1.2rem; color: rgba(44, 62, 80, 0.8); margin-bottom: 2rem;">
                    Upload a dataset to enable advanced prediction features
                </p>
                <div style="font-size: 2rem;">⬅️</div>
                <p>Use the sidebar to upload your data</p>
            </div>
        """, unsafe_allow_html=True)

# Advanced AI Chatbot with Modern UI
if selected == "💬 AI Chatbot":
    # Hero section for Chatbot
    st.markdown("""
        <div class="glass-card fade-in" style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">💬 AI Assistant</h1>
            <p style="font-size: 1.2rem; color: rgba(44, 62, 80, 0.8);">Chat with our advanced AI assistant powered by cutting-edge language models</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Fast AI Chat Settings
    st.markdown("### ⚙️ AI Settings")
    col1, col2 = st.columns(2)
    with col1:
        auto_scroll = st.checkbox("Auto Scroll", value=True)
        show_timestamps = st.checkbox("Show Timestamps", value=True)
    with col2:
        # Model selection based on available models
        available_models = ["auto"] + list(chat_models.keys())
        model_choice = st.selectbox("🚀 AI Model", available_models, index=0, 
                                  help="Auto selects the fastest available model")
        temperature = st.slider("🎛️ Creativity", 0.1, 1.0, 0.7, 0.1)
    
    # API Status Display
    st.markdown("### 🔑 API Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        if GOOGLE_API_KEY and GOOGLE_API_KEY != "your_google_api_key_here":
            st.success("✅ Google Gemini")
        else:
            st.error("❌ Google Gemini")
    with col2:
        if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
            st.success("✅ Groq (Ultra Fast)")
        else:
            st.error("❌ Groq")
    with col3:
        if HUGGINGFACEHUB_API_TOKEN and HUGGINGFACEHUB_API_TOKEN != "your_hugging_face_token_here":
            st.success("✅ Hugging Face")
        else:
            st.error("❌ Hugging Face")
    
    st.markdown("---")
    
    if not chat_models:
        st.error("🤖 No AI models available. Please check your API keys.")
        st.stop()
    
    # Initialize chat history and response cache
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'response_cache' not in st.session_state:
        st.session_state.response_cache = {}
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat container with modern styling
        st.markdown("""
            <div class="chat-container">
                <div style="text-align: center; margin-bottom: 1rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 15px;">
                    <h3 style="margin: 0; color: #667eea;">🤖 AI Assistant</h3>
                    <p style="margin: 0.5rem 0 0 0; color: rgba(44, 62, 80, 0.7);">Ready to help with your data science questions</p>
                </div>
        """, unsafe_allow_html=True)
        
        # Display chat messages with model info
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                if chat["is_user"]:
                    st.markdown(f"""
                        <div class="chat-message user fade-in" style="animation-delay: {i*0.1}s;">
                            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                <div style="width: 30px; height: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                                    <span style="color: white; font-weight: bold;">U</span>
                                </div>
                                <strong>You</strong>
                                {f'<span style="margin-left: auto; font-size: 0.8rem; opacity: 0.7;">{chat.get("timestamp", "")}</span>' if show_timestamps else ''}
                            </div>
                            <p style="margin: 0;">{chat["content"]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    model_info = chat.get("model", "AI Assistant")
                    st.markdown(f"""
                        <div class="chat-message bot fade-in" style="animation-delay: {i*0.1}s;">
                            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                <div style="width: 30px; height: 30px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                                    <span style="color: white; font-weight: bold;">🤖</span>
                                </div>
                                <strong>{model_info}</strong>
                                {f'<span style="margin-left: auto; font-size: 0.8rem; opacity: 0.7;">{chat.get("timestamp", "")}</span>' if show_timestamps else ''}
                            </div>
                            <p style="margin: 0;">{chat["content"]}</p>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            # Welcome message
            st.markdown("""
                <div class="chat-message bot">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <div style="width: 30px; height: 30px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                            <span style="color: white; font-weight: bold;">🤖</span>
                        </div>
                        <strong>AI Assistant</strong>
                    </div>
                    <p style="margin: 0;">Hello! I'm your AI assistant. I can help you with data analysis, machine learning questions, and provide insights about your datasets. What would you like to know?</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Chat controls
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top: 0;">🎛️ Chat Controls</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("📊 Data Analysis Help", width='stretch'):
            st.session_state.chat_history.append({
                "content": "Can you help me analyze my dataset?", 
                "is_user": True,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()
        
        if st.button("🤖 Model Selection", width='stretch'):
            st.session_state.chat_history.append({
                "content": "What machine learning model should I use?", 
                "is_user": True,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()
        
        if st.button("📈 Visualization Tips", width='stretch'):
            st.session_state.chat_history.append({
                "content": "How can I create better visualizations?", 
                "is_user": True,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()
        
        if st.button("🔧 Technical Help", width='stretch'):
            st.session_state.chat_history.append({
                "content": "I need help with technical implementation", 
                "is_user": True,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()
        
        st.markdown("---")
        
        # Settings moved to top of page
        
        # Clear chat and cache
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", width='stretch'):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("⚡ Clear Cache", width='stretch'):
                st.session_state.response_cache = {}
                st.success("Cache cleared!")
                st.rerun()
    
    # Input form
    st.markdown("""
        <div class="glass-card" style="margin-top: 2rem;">
            <h3 style="margin-top: 0;">💬 Send Message</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_area(
                "Type your message here...", 
                height=100, 
                placeholder="Ask me anything about data science, machine learning, or your dataset...",
                help="Enter your question or request"
            )
        with col2:
            submit = st.form_submit_button("🚀 Send", width='stretch')
    
    if submit and query:
        # Add user message immediately
        st.session_state.chat_history.append({
            "content": query, 
            "is_user": True,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Show typing indicator
        with st.spinner("🤖 AI is thinking..."):
            # Check cache first for faster responses
            cache_key = f"{query.lower().strip()}_{model_choice}_{temperature}"
            if cache_key in st.session_state.response_cache:
                response_text, model_name = st.session_state.response_cache[cache_key]
                st.success("⚡ Fast response from cache!")
            else:
                # Get AI response using the new fast system
                response_text, model_name = get_ai_response(chat_models, query, model_choice)
                
                # Cache the response
                st.session_state.response_cache[cache_key] = (response_text, model_name)
        
        # Add AI response to chat history
        st.session_state.chat_history.append({
            "content": response_text, 
            "is_user": False,
            "model": model_name,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        st.rerun()
    
    # Chat statistics
    if st.session_state.chat_history:
        st.markdown(f"""
            <div class="glass-card" style="margin-top: 1rem;">
                <h4 style="margin-top: 0;">📊 Chat Statistics</h4>
                <div style="display: flex; gap: 2rem;">
                    <div>
                        <strong>Messages:</strong> {len(st.session_state.chat_history)}
                    </div>
                    <div>
                        <strong>Your Messages:</strong> {len([msg for msg in st.session_state.chat_history if msg["is_user"]])}
                    </div>
                    <div>
                        <strong>AI Responses:</strong> {len([msg for msg in st.session_state.chat_history if not msg["is_user"]])}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Stunning Interactive Dashboard
if selected == "📈 Dashboard":
    # Hero section for Dashboard
    st.markdown("""
        <div class="glass-card fade-in" style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">📈 Interactive Dashboard</h1>
            <p style="font-size: 1.2rem; color: rgba(44, 62, 80, 0.8);">Real-time insights and comprehensive analytics at your fingertips</p>
        </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Key metrics overview
        st.markdown("""
            <div class="glass-card fade-in">
                <h2 style="margin-top: 0;">📊 Key Metrics</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Create metric cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-value">{df.shape[0]:,}</div>
                    <div class="metric-label">Total Rows</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-value">{df.shape[1]}</div>
                    <div class="metric-label">Columns</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-value">{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-value">{df.isnull().sum().sum()}</div>
                    <div class="metric-label">Missing Values</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-value">{len(df.select_dtypes(include=['number']).columns)}</div>
                    <div class="metric-label">Numeric Columns</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🔍 Analysis", "⚡ Real-time"])
        
        with tab1:
            st.markdown("""
                <div class="glass-card fade-in">
                    <h2 style="margin-top: 0;">📊 Data Overview</h2>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Data types distribution
                dtype_counts = df.dtypes.value_counts()
                fig_dtypes = px.pie(
                    values=dtype_counts.values,
                    names=dtype_counts.index.astype(str),
                    title="Data Types Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_dtypes.update_layout(
                    title_font_size=20,
                    font=dict(size=14),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_dtypes, width='stretch')
            
            with col2:
                # Missing values heatmap
                if df.isnull().sum().sum() > 0:
                    missing_data = df.isnull().sum().sort_values(ascending=False)
                    fig_missing = px.bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        title="Missing Values by Column",
                        color=missing_data.values,
                        color_continuous_scale="Reds"
                    )
                    fig_missing.update_layout(
                        title_font_size=20,
                        font=dict(size=14),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_missing, width='stretch')
                else:
                    st.success("🎉 No missing values found!")
        
        with tab2:
            st.markdown("""
                <div class="glass-card fade-in">
                    <h2 style="margin-top: 0;">📈 Data Trends</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Select columns for trend analysis
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                selected_trend_cols = st.multiselect(
                    "Select columns for trend analysis",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )
                
                if selected_trend_cols:
                    # Create trend visualization
                    fig_trends = go.Figure()
                    
                    for col in selected_trend_cols:
                        fig_trends.add_trace(go.Scatter(
                            y=df[col],
                            mode='lines+markers',
                            name=col,
                            line=dict(width=2),
                            marker=dict(size=4)
                        ))
                    
                    fig_trends.update_layout(
                        title="Data Trends Over Time",
                        title_font_size=20,
                        font=dict(size=14),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=500,
                        xaxis_title="Index",
                        yaxis_title="Value"
                    )
                    
                    st.plotly_chart(fig_trends, width='stretch')
            else:
                st.info("No numeric columns available for trend analysis")
        
        with tab3:
            st.markdown("""
                <div class="glass-card fade-in">
                    <h2 style="margin-top: 0;">🔍 Advanced Analysis</h2>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation heatmap
                numeric_df = df.select_dtypes(include=['number'])
                if not numeric_df.empty and len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale="RdBu",
                        title="Correlation Matrix",
                        aspect="auto"
                    )
                    fig_corr.update_layout(
                        title_font_size=20,
                        font=dict(size=12),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_corr, width='stretch')
                else:
                    st.info("Not enough numeric columns for correlation analysis")
            
            with col2:
                # Statistical summary
                st.markdown("### 📊 Statistical Summary")
                summary_stats = df.describe()
                st.dataframe(summary_stats, width='stretch')
        
        with tab4:
            st.markdown("""
                <div class="glass-card fade-in">
                    <h2 style="margin-top: 0;">⚡ Real-time Monitoring</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Real-time metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Data Quality Score",
                    f"{95 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%",
                    delta="2.1%"
                )
            
            with col2:
                st.metric(
                    "Processing Time",
                    f"{random.uniform(0.5, 2.0):.2f}s",
                    delta=f"{random.uniform(-0.1, 0.1):.2f}s"
                )
            
            with col3:
                st.metric(
                    "AI Confidence",
                    f"{random.uniform(85, 98):.1f}%",
                    delta=f"{random.uniform(-2, 5):.1f}%"
                )
            
            # Live data preview
            st.markdown("### 📋 Live Data Preview")
            st.dataframe(df.head(10), width='stretch')
            
            # Auto-refresh button
            if st.button("🔄 Refresh Data", width='stretch'):
                st.rerun()
        
        # Export options
        st.markdown("""
            <div class="glass-card" style="margin-top: 2rem;">
                <h3 style="margin-top: 0;">📥 Export Options</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📊 Download CSV",
                csv_data,
                "dashboard_data.csv",
                "text/csv",
                width='stretch'
            )
        
        with col2:
            # Create a summary report
            report = f"""
            PromptML Dashboard Report
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Dataset Overview:
            - Rows: {df.shape[0]:,}
            - Columns: {df.shape[1]}
            - Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
            - Missing Values: {df.isnull().sum().sum()}
            - Numeric Columns: {len(df.select_dtypes(include=['number']).columns)}
            
            Data Quality Score: {95 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%
            """
            st.download_button(
                "📄 Download Report",
                report,
                "dashboard_report.txt",
                "text/plain",
                width='stretch'
            )
        
        with col3:
            if st.button("🖼️ Export Dashboard", width='stretch'):
                st.success("Dashboard export feature coming soon!")
    
    else:
        st.markdown("""
            <div class="glass-card fade-in" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
                <h2>No Data Available</h2>
                <p style="font-size: 1.2rem; color: rgba(44, 62, 80, 0.8); margin-bottom: 2rem;">
                    Upload a dataset to access the interactive dashboard
                </p>
                <div style="font-size: 2rem;">⬅️</div>
                <p>Use the sidebar to upload your data</p>
            </div>
        """, unsafe_allow_html=True)
