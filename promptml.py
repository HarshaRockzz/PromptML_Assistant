import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv, find_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from streamlit_chat import message
from streamlit_elements import elements, mui, html, dashboard

# Load environment variables
load_dotenv(find_dotenv())

# Set Streamlit page config
st.set_page_config(page_title="PromptML: NextGen AI Assistant", layout="wide")

# Sidebar Navigation
with st.sidebar:
    st.header("🚀 AI Assistant Navigation")
    page = st.radio("Choose a Section", ["📊 EDA", "🤖 Model Selection", "🔮 Predictions", "💬 AI Chatbot", "📈 Interactive Dashboard"])
    temperature = st.slider("LLM Temperature", 0.01, 1.0, 0.1, 0.01)
    uploaded_files = st.file_uploader("Upload Data (CSV/Excel)", type=["csv", "xlsx"], accept_multiple_files=True)

# Load Data
if uploaded_files:
    dfs = [pd.read_csv(file, low_memory=False) if file.name.endswith(".csv") else pd.read_excel(file) for file in uploaded_files]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
else:
    df = None

# Local Model Setup
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

llm = HuggingFacePipeline.from_model_id(
    model_id=model_name, tokenizer=tokenizer, model=model
)

if page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")
    if df is not None:
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        tab1, tab2, tab3 = st.tabs(["Overview", "EDA Steps", "Custom Analysis"])
        
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df.head())
            st.subheader("Statistical Summary")
            st.write(df.describe())
            st.subheader("Distribution Plot")
            fig = px.histogram(df, x=df.columns[0])
            st.plotly_chart(fig)
        
        with tab2:
            steps = llm("What are the steps of EDA")
            st.write(steps)
            st.write(pandas_agent.run("Summarize the dataset"))
        
        with tab3:
            user_query = st.text_input("Ask about your data:")
            if user_query:
                result = pandas_agent.run(user_query)
                st.write(result)
    else:
        st.warning("Please upload a dataset to proceed.")

elif page == "🤖 Model Selection":
    st.header("🤖 AI-Powered Model Selection")
    if df is not None:
        model_recommendation = llm("Recommend the best machine learning model for this dataset")
        st.write(model_recommendation)
    else:
        st.warning("Please upload a dataset to get model recommendations.")

elif page == "🔮 Predictions":
    st.header("🔮 AI Predictions")
    st.write("Upload test data and receive real-time AI predictions.")
    if df is not None:
        user_query = st.text_input("Describe your prediction task:")
        if user_query:
            prediction = llm(f"Based on the dataset, make predictions for: {user_query}")
            st.write(prediction)
    else:
        st.warning("Please upload a dataset first.")

elif page == "💬 AI Chatbot":
    st.header("💬 Conversational AI Chatbot")
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["Hello! How can I assist you?"]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    
    chat_llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        temperature=temperature
    )
    chat_model = ChatHuggingFace(llm=chat_llm)
    
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    
    sys_msg_template = SystemMessagePromptTemplate.from_template(
        template="Answer truthfully using provided context. If unsure, say 'I don’t know'."
    )
    human_msg_temp = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages(
        [sys_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_temp]
    )
    conversation = ConversationChain(
        memory=st.session_state.buffer_memory, prompt=prompt_template, llm=chat_llm, verbose=True
    )
    
    query = st.text_input("Your question:", key="chat_input")
    if query:
        with st.spinner("Thinking..."):
            response = conversation.predict(input=query)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
    
    for i in range(len(st.session_state['responses'])):
        message(st.session_state['responses'][i], key=str(i))
        if i < len(st.session_state['requests']):
            message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

elif page == "📈 Interactive Dashboard":
    st.header("📈 Interactive Data Dashboard")
    if df is not None:
        with elements("dashboard"):
            mui.Typography("Dynamic Dashboard", variant="h4")
            with mui.Grid(container=True, spacing=2):
                with mui.Grid(item=True, xs=6):
                    mui.Paper(
                        mui.Typography("Data Summary", variant="h6"),
                        html.div(str(df.describe().to_html()), style={"overflow": "auto", "maxHeight": "300px"})
                    )
                with mui.Grid(item=True, xs=6):
                    mui.Paper(
                        mui.Typography("Column Names", variant="h6"),
                        html.ul([html.li(col) for col in df.columns])
                    )
    else:
        st.warning("Please upload a dataset to view the dashboard.")
