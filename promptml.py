import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from streamlit_chat import message
import plotly.graph_objects as go
from datetime import datetime
import psutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set page config
st.set_page_config(
    page_title="PromptML: NextGen AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Validate imports and token
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in .env file. Please set it and restart the app.")
    st.stop()

# Advanced Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150", caption="PromptML")
    theme = st.toggle("Dark Mode", value=False)
    selected = st.sidebar.selectbox(
        "Navigation", ["EDA", "Model Selection", "Predictions", "AI Chatbot", "Dashboard"], index=0
    )
    temperature = st.slider("AI Temperature", 0.01, 1.0, 0.1, 0.01)
    uploaded_files = st.file_uploader("Upload Data", type=["csv", "xlsx"], accept_multiple_files=True)
    st.button("Refresh Data", key="refresh")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; color: #212121; }
    .stButton>button { background-color: #0288d1; color: white; border-radius: 12px; padding: 10px 20px; }
    .stTextInput>div>input { border-radius: 12px; padding: 8px; border: 1px solid #0288d1; background-color: #ffffff; }
    </style>
""", unsafe_allow_html=True)
if theme:
    st.markdown("""
        <style>
        .main { background-color: #212121; color: #e0e0e0; }
        .stTextInput>div>input { background-color: #424242; color: #e0e0e0; }
        </style>
    """, unsafe_allow_html=True)

# Load Data
if uploaded_files:
    with st.spinner("Loading data..."):
        dfs = [pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f) for f in uploaded_files]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        st.success("Data loaded successfully!")
else:
    df = None

# Model Setup
@st.cache_resource
def load_generative_model():
    try:
        model_name = "distilgpt2"
        st.write(f"Attempting to load model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACEHUB_API_TOKEN)
        st.write("Tokenizer loaded successfully.")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGINGFACEHUB_API_TOKEN)
        st.write("Model loaded successfully.")
        
        pipeline_instance = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer
        )
        
        return HuggingFacePipeline(pipeline=pipeline_instance)
    except Exception as e:
        st.error(f"Generative model loading failed: {str(e)}")
        return None

@st.cache_resource
def load_classification_pipeline():
    try:
        model_name = "distilbert-base-uncased"
        return pipeline("text-classification", model=model_name, tokenizer=model_name, token=HUGGINGFACEHUB_API_TOKEN)
    except Exception as e:
        st.error(f"Classification pipeline loading failed: {str(e)}")
        return None

llm = load_generative_model()
classifier = load_classification_pipeline()

# EDA Page
if selected == "EDA":
    st.title("📊 Exploratory Data Analysis")
    if df is not None:
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True) if llm else None
        tab1, tab2, tab3 = st.tabs(["Overview", "Health", "Visualizations"])
        
        with tab1:
            st.subheader("Data Overview")
            st.dataframe(df.head(), use_container_width=True)
            with st.expander("Statistical Summary"):
                st.write(df.describe())
        
        with tab2:
            st.subheader("Data Health")
            missing = df.isnull().sum()
            fig_missing = px.pie(names=missing.index, values=missing.values, title="Missing Values", hole=0.3)
            st.plotly_chart(fig_missing)
        
        with tab3:
            st.subheader("Interactive Visualizations")
            col1, col2 = st.columns([1, 2])
            with col1:
                viz_type = st.selectbox("Chart Type", ["Histogram", "Scatter", "Box", "Correlation"])
                column = st.selectbox("Select Column", df.columns)
            with col2:
                if viz_type == "Histogram":
                    fig = px.histogram(df, x=column)
                elif viz_type == "Scatter":
                    y_col = st.selectbox("Y-axis", df.columns)
                    fig = px.scatter(df, x=column, y=y_col, trendline="ols")
                elif viz_type == "Box":
                    fig = px.box(df, y=column)
                else:
                    fig = px.imshow(df.corr(), text_auto=True)
                st.plotly_chart(fig)

# Model Selection with Training
elif selected == "Model Selection":
    st.title("🤖 Model Selection & Training")
    if df is not None:
        with st.form("model_form"):
            problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])
            target = st.selectbox("Target Variable", df.columns)
            train_model = st.checkbox("Train a Model")
            submitted = st.form_submit_button("Analyze")
        
        if submitted:
            if llm:
                recommendation = llm(f"Recommend top 3 ML models for {problem_type} with target {target}")
                st.success(recommendation)
            if train_model and problem_type == "Classification":
                with st.spinner("Training model..."):
                    X = df.drop(columns=[target]).select_dtypes(include=['float64', 'int64'])
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = RandomForestClassifier(n_estimators=100)
                    model.fit(X_train, y_train)
                    accuracy = accuracy_score(y_test, model.predict(X_test))
                    st.success(f"Model trained! Accuracy: {accuracy:.2f}")
                    st.session_state['trained_model'] = model
    else:
        st.warning("Upload a dataset first!")

# Predictions
elif selected == "Predictions":
    st.title("🔮 Predictions")
    if df is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            model_choice = st.selectbox("Select Model", ["Trained Model", "Pre-trained Classifier"])
            predict_button = st.button("Predict")
        
        with col2:
            if predict_button:
                if model_choice == "Trained Model" and 'trained_model' in st.session_state:
                    X = df.select_dtypes(include=['float64', 'int64'])
                    predictions = st.session_state['trained_model'].predict(X)
                    st.write("Predictions:", predictions)
                    fig = go.Figure(data=go.Scatter(y=predictions, mode="lines+markers"))
                    st.plotly_chart(fig)
                elif model_choice == "Pre-trained Classifier" and classifier:
                    text_col = st.selectbox("Select Text Column", df.columns)
                    predictions = [classifier(text)[0]['label'] for text in df[text_col].head(10)]
                    st.write("Predictions (Top 10):", predictions)
                else:
                    st.error("No suitable model available.")
    else:
        st.warning("Upload a dataset first!")

# AI Chatbot with Memory
elif selected == "AI Chatbot":
    st.title("💬 AI Chatbot")
    if llm:
        chat_llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            temperature=temperature,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
        )
        chat_model = ChatHuggingFace(llm=chat_llm)
        memory = ConversationBufferWindowMemory(k=5)
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        chat_container = st.container(height=400)
        with chat_container:
            for chat in st.session_state.chat_history:
                message(chat["content"], is_user=chat["is_user"])
        
        with st.form("chat_form", clear_on_submit=True):
            query = st.text_area("Ask me anything:", height=100)
            submit, clear, export = st.columns(3)
            with submit:
                st.form_submit_button("Send")
            with clear:
                st.form_submit_button("Clear Chat")
            with export:
                st.form_submit_button("Export Chat")
            
            if submit and query:
                response = chat_model.predict(query)
                st.session_state.chat_history.append({"content": query, "is_user": True})
                st.session_state.chat_history.append({"content": response, "is_user": False})
                memory.save_context({"input": query}, {"output": response})
                st.rerun()
            if clear:
                st.session_state.chat_history = []
                memory.clear()
                st.rerun()
            if export:
                chat_json = json.dumps(st.session_state.chat_history)
                st.download_button("Download Chat", chat_json, "chat_history.json")
    else:
        st.warning("Chatbot disabled due to model loading failure.")

# Dashboard with Real-time Metrics
elif selected == "Dashboard":
    st.title("📈 Interactive Dashboard")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
        with col2:
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent}%")
            st.metric("Last Updated", datetime.now().strftime('%H:%M:%S'))
        
        fig = px.line(df.select_dtypes(include=['float64', 'int64']))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload a dataset to view dashboard!")
