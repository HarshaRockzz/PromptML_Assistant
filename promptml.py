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
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stTextInput>div>input {border-radius: 8px;}
    .sidebar .sidebar-content {background-color: #ffffff; border-right: 1px solid #ddd;}
    </style>
""", unsafe_allow_html=True)

# Set page config with custom icon
st.set_page_config(
    page_title="PromptML: NextGen AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Sidebar with option menu
with st.sidebar:
    st.image("https://via.placeholder.com/150", caption="PromptML")
    selected = option_menu(
        menu_title="Navigation",
        options=["EDA", "Model Selection", "Predictions", "AI Chatbot", "Dashboard"],
        icons=["bar-chart", "cpu", "graph-up", "chat", "speedometer2"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#4CAF50", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )
    temperature = st.slider("AI Temperature", 0.01, 1.0, 0.1, 0.01)
    uploaded_files = st.file_uploader("Upload Data", type=["csv", "xlsx"], accept_multiple_files=True)

# Load Data with Progress Bar
if uploaded_files:
    with st.spinner("Loading data..."):
        progress_bar = st.progress(0)
        dfs = []
        for i, file in enumerate(uploaded_files):
            dfs.append(pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file))
            progress_bar.progress((i + 1) / len(uploaded_files))
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
else:
    df = None

# Model Setup with Error Handling
@st.cache_resource
def load_model():
    try:
        model_name = "mistralai/Mistral-7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        return HuggingFacePipeline.from_model_id(model_id=model_name, tokenizer=tokenizer, model=model)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

llm = load_model()

# EDA Page with Advanced Visualizations
if selected == "EDA":
    st.title("📊 Exploratory Data Analysis")
    if df is not None:
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        
        # Multi-column layout
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Overview")
            st.dataframe(df.head(), use_container_width=True)
            with st.expander("Statistical Summary"):
                st.write(df.describe())
        
        with col2:
            st.subheader("Data Health")
            missing = df.isnull().sum()
            fig_missing = px.pie(names=missing.index, values=missing.values, title="Missing Values")
            st.plotly_chart(fig_missing)
        
        # Interactive Visualizations
        st.subheader("Interactive Visualizations")
        viz_type = st.selectbox("Chart Type", ["Histogram", "Scatter", "Box", "Correlation"])
        column = st.selectbox("Select Column", df.columns)
        
        if viz_type == "Histogram":
            fig = px.histogram(df, x=column, marginal="box")
        elif viz_type == "Scatter":
            y_col = st.selectbox("Y-axis", df.columns)
            fig = px.scatter(df, x=column, y=y_col, trendline="ols")
        elif viz_type == "Box":
            fig = px.box(df, y=column)
        else:
            fig = px.imshow(df.corr(), text_auto=True)
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
            with st.spinner("Analyzing..."):
                recommendation = llm(f"Recommend top 3 ML models for {problem_type} with target {target}")
                st.success(recommendation)
                st.balloons()
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
                with st.spinner("Generating predictions..."):
                    predictions = llm(f"Predict {model_choice} outcomes for the dataset")
                    st.write(predictions)
                    fig = go.Figure(data=go.Scatter(y=[float(x) for x in predictions.split() if x.isdigit()]))
                    st.plotly_chart(fig)
    else:
        st.warning("Upload a dataset first!")

# Enhanced Chatbot with Rich UI
elif selected == "AI Chatbot":
    st.title("💬 AI Chatbot")
    chat_llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        temperature=temperature
    )
    chat_model = ChatHuggingFace(llm=chat_llm)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            message(chat["content"], is_user=chat["is_user"], avatar_style="bottts" if not chat["is_user"] else "adventurer")
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        query = st.text_area("Ask me anything:", height=100)
        submit = st.form_submit_button("Send")
    
    if submit and query:
        with st.spinner("Responding..."):
            response = chat_model.predict(query)
            st.session_state.chat_history.append({"content": query, "is_user": True})
            st.session_state.chat_history.append({"content": response, "is_user": False})
            st.rerun()

# Advanced Dashboard
elif selected == "Dashboard":
    st.title("📈 Interactive Dashboard")
    if df is not None:
        with elements("advanced_dashboard"):
            with mui.Stack(spacing=2):
                mui.Card(
                    mui.CardContent(
                        mui.Typography("Dataset Metrics", variant="h5"),
                        html.div(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                    )
                )
                mui.Card(
                    mui.CardContent(
                        mui.Typography("Real-time Insights", variant="h5"),
                        html.div(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
                    )
                )
    else:
        st.warning("Upload a dataset to view dashboard!")
