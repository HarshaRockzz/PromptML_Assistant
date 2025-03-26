import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from streamlit_chat import message
from streamlit_elements import elements, mui, html, dashboard
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from datetime import datetime

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in .env file. Please set it and restart the app.")

# Set page config with custom icon (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="PromptML: NextGen AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Sidebar with option menu and theme toggle
with st.sidebar:
    st.image("https://via.placeholder.com/150", caption="PromptML")
    theme = st.toggle("Dark Mode", value=False)
    
    selected = option_menu(
        menu_title="Navigation",
        options=["EDA", "Model Selection", "Predictions", "AI Chatbot", "Dashboard"],
        icons=["bar-chart", "cpu", "graph-up", "chat", "speedometer2"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "10px", "background-color": "#e0e0e0" if not theme else "#2a2a2a"},
            "icon": {"color": "#0288d1", "font-size": "22px"},
            "nav-link": {"font-size": "18px", "text-align": "left", "margin": "5px", "--hover-color": "#b0bec5", "color": "#212121" if not theme else "#e0e0e0"},
            "nav-link-selected": {"background-color": "#0288d1", "color": "white"},
        }
    )
    temperature = st.slider("AI Temperature", 0.01, 1.0, 0.1, 0.01)
    uploaded_files = st.file_uploader("Upload Data", type=["csv", "xlsx"], accept_multiple_files=True)

# Custom CSS for enhanced UI with improved colors
st.markdown("""
    <style>
    /* Light Mode */
    .main {
        background-color: #f5f5f5;
        color: #212121;
    }
    .stButton>button {
        background-color: #0288d1;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0277bd;
    }
    .stTextInput>div>input {
        border-radius: 12px;
        padding: 8px;
        border: 1px solid #0288d1;
        background-color: #ffffff;
        color: #212121;
    }
    .sidebar .sidebar-content {
        background-color: #e0e0e0;
        border-right: 2px solid #b0bec5;
        padding: 20px;
    }
    .stSlider > div > div > div {
        background-color: #0288d1;
    }
    .stAlert {
        background-color: #ffebee !important;
        color: #d32f2f !important;
        border: 1px solid #d32f2f;
        border-radius: 8px;
    }
    .stWarning {
        background-color: #fff3e0 !important;
        color: #f57c00 !important;
        border: 1px solid #f57c00;
        border-radius: 8px;
    }
    .stSuccess {
        background-color: #e8f5e9 !important;
        color: #2e7d32 !important;
        border: 1px solid #2e7d32;
        border-radius: 8px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0288d1;
    }
    </style>
""", unsafe_allow_html=True)

# Apply dark mode styles if toggled
if theme:
    st.markdown("""
        <style>
        .main {
            background-color: #212121;
            color: #e0e0e0;
        }
        .stTextInput>div>input {
            background-color: #424242;
            color: #e0e0e0;
            border: 1px solid #0288d1;
        }
        .stAlert {
            background-color: #d32f2f !important;
            color: #ffffff !important;
            border: 1px solid #b71c1c;
        }
        .stWarning {
            background-color: #ff9800 !important;
            color: #ffffff !important;
            border: 1px solid #f57c00;
        }
        .stSuccess {
            background-color: #4caf50 !important;
            color: #ffffff !important;
            border: 1px solid #388e3c;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #4fc3f7;
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

# Model Setup with Error Handling
@st.cache_resource
def load_model():
    try:
        model_name = "mistralai/Mistral-7B-v0.1"
        st.write(f"Attempting to load model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACEHUB_API_TOKEN)
        st.write("Tokenizer loaded successfully.")
        # Temporarily remove device_map to bypass accelerate requirement for debugging
        model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGINGFACEHUB_API_TOKEN)
        st.write("Model loaded successfully.")
        return HuggingFacePipeline.from_model_id(model_id=model_name, tokenizer=tokenizer, model=model)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}. Please check your Hugging Face token and network connection.")
        return None

llm = load_model()

# Check if model loaded successfully
if llm is None:
    st.warning("AI features are disabled due to model loading failure. Basic functionality is still available.")

# EDA Page with Advanced Visualizations
if selected == "EDA":
    st.title("📊 Exploratory Data Analysis")
    if df is not None:
        if llm is not None:
            pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        else:
            st.warning("EDA agent is disabled due to model loading failure.")
        
        # Multi-column layout with tabs
        tab1, tab2, tab3 = st.tabs(["Overview", "Health", "Visualizations"])
        
        with tab1:
            st.subheader("Data Overview")
            st.dataframe(df.head(), use_container_width=True, height=300)
            with st.expander("Statistical Summary", expanded=False):
                st.write(df.describe())
        
        with tab2:
            st.subheader("Data Health")
            missing = df.isnull().sum()
            fig_missing = px.pie(names=missing.index, values=missing.values, title="Missing Values", hole=0.3)
            st.plotly_chart(fig_missing, use_container_width=True)
        
        with tab3:
            st.subheader("Interactive Visualizations")
            col1, col2 = st.columns([1, 2])
            with col1:
                viz_type = st.selectbox("Chart Type", ["Histogram", "Scatter", "Box", "Correlation"])
                column = st.selectbox("Select Column", df.columns)
            with col2:
                if viz_type == "Histogram":
                    fig = px.histogram(df, x=column, marginal="box", color_discrete_sequence=["#0288d1"])
                elif viz_type == "Scatter":
                    y_col = st.selectbox("Y-axis", df.columns)
                    fig = px.scatter(df, x=column, y=y_col, trendline="ols", color_discrete_sequence=["#0288d1"])
                elif viz_type == "Box":
                    fig = px.box(df, y=column, color_discrete_sequence=["#0288d1"])
                else:
                    fig = px.imshow(df.corr(), text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)

# Model Selection with Advanced Features
elif selected == "Model Selection":
    st.title("🤖 Model Selection")
    if df is not None:
        with st.form("model_form"):
            problem_type = st.selectbox("Problem Type", ["Classification", "Regression", "Clustering"])
            target = st.selectbox("Target Variable", df.columns)
            submitted = st.form_submit_button("Get Recommendations")
        
        if submitted:
            if llm is not None:
                with st.spinner("Analyzing..."):
                    recommendation = llm(f"Recommend top 3 ML models for {problem_type} with target {target}")
                    st.success(recommendation)
                    st.balloons()
                    st.download_button("Download Recommendations", recommendation, "recommendations.txt")
            else:
                st.error("Cannot generate recommendations because the AI model failed to load.")
    else:
        st.warning("Upload a dataset first!")

# Predictions with Real-time Updates
elif selected == "Predictions":
    st.title("🔮 Predictions")
    if df is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "Neural Network"])
            predict_button = st.button("Predict")
        
        with col2:
            if predict_button:
                if llm is not None:
                    with st.spinner("Generating predictions..."):
                        predictions = llm(f"Predict {model_choice} outcomes for the dataset")
                        st.write(predictions)
                        fig = go.Figure(data=go.Scatter(y=[float(x) for x in predictions.split() if x.replace('.', '', 1).isdigit()],
                                                        mode="lines+markers", line=dict(color="#0288d1")))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Cannot generate predictions because the AI model failed to load.")
    else:
        st.warning("Upload a dataset first!")

# Enhanced Chatbot with Rich UI
elif selected == "AI Chatbot":
    st.title("💬 AI Chatbot")
    if llm is not None:
        chat_llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            temperature=temperature,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
        )
        chat_model = ChatHuggingFace(llm=chat_llm)
    else:
        st.warning("Chatbot is disabled due to model loading failure.")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat container with scroll
    chat_container = st.container(height=400)
    with chat_container:
        for chat in st.session_state.chat_history:
            message(chat["content"], is_user=chat["is_user"], avatar_style="bottts" if not chat["is_user"] else "adventurer")
    
    # Input form with advanced features
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_area("Ask me anything:", height=100, placeholder="Type your question here...")
        with col2:
            submit = st.form_submit_button("Send")
            clear = st.form_submit_button("Clear Chat")
        
        if clear:
            st.session_state.chat_history = []
            st.rerun()
    
    if submit and query:
        if llm is not None:
            with st.spinner("Responding..."):
                response = chat_model.predict(query)
                st.session_state.chat_history.append({"content": query, "is_user": True})
                st.session_state.chat_history.append({"content": response, "is_user": False})
                st.rerun()
        else:
            st.error("Cannot respond because the AI model failed to load.")

# Advanced Dashboard
elif selected == "Dashboard":
    st.title("📈 Interactive Dashboard")
    if df is not None:
        with elements("advanced_dashboard"):
            layout = [
                dashboard.Item("metrics", 0, 0, 2, 1),
                dashboard.Item("insights", 2, 0, 2, 1),
                dashboard.Item("plot", 0, 1, 4, 2),
            ]
            with dashboard.Grid(layout):
                mui.Card(
                    mui.CardContent(
                        mui.Typography("Dataset Metrics", variant="h5"),
                        html.div(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                    ),
                    key="metrics"
                )
                mui.Card(
                    mui.CardContent(
                        mui.Typography("Real-time Insights", variant="h5"),
                        html.div(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
                    ),
                    key="insights"
                )
                mui.Card(
                    mui.CardContent(
                        mui.Typography("Data Trend", variant="h5"),
                        components.html(px.line(df.select_dtypes(include=['float64', 'int64'])).to_html(), height=300)
                    ),
                    key="plot"
                )
    else:
        st.warning("Upload a dataset to view dashboard!")
