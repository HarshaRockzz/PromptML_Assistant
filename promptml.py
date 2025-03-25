import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.agents import initialize_agent, Tool
from langchain_experimental.utilities import PythonREPL
from langchain.agents.agent_types import AgentType
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
import wikipedia
from streamlit_elements import elements, mui, html
from important_functions import *

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Set wide layout
st.set_page_config(layout="wide")

# Title and Header
st.title("Advanced AI Data Scientist Assistant")
st.markdown("Explore your data with cutting-edge tools and a smart chatbot!")

# Sidebar for navigation and settings
with st.sidebar:
    st.header("Settings & Navigation")
    page = st.selectbox("Choose a Page", ["Data Analysis", "Dashboard", "Chatbot"])
    st.slider("Temperature (for LLM)", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="temp")
    uploaded_files = st.file_uploader("Upload CSV Files", type="csv", accept_multiple_files=True)

# Main content based on selected page
if page == "Data Analysis":
    st.header("Data Analysis Section")
    if uploaded_files:
        dfs = [pd.read_csv(file, low_memory=False) for file in uploaded_files]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        
        # Initialize LLM
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-v0.1",
            model_kwargs={'temperature': st.session_state.temp},
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

        # Tabs for different analysis sections
        tab1, tab2, tab3 = st.tabs(["Overview", "EDA Steps", "Custom Analysis"])
        
        with tab1:
            st.subheader("Data Overview")
            st.write(df.head())
            st.write(df.describe())
        
        with tab2:
            st.subheader("Exploratory Data Analysis")
            with st.expander("EDA Steps"):
                steps = llm('What are the steps of EDA')
                st.write(steps)
            st.write(pandas_agent.run("Summarize the dataset"))

        with tab3:
            st.subheader("Custom Analysis")
            user_query = st.text_input("Ask about your data:")
            if user_query:
                result = pandas_agent.run(user_query)
                st.write(result)

elif page == "Dashboard":
    st.header("Interactive Dashboard")
    if uploaded_files:
        dfs = [pd.read_csv(file, low_memory=False) for file in uploaded_files]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # Draggable and resizable dashboard using streamlit-elements
        with elements("dashboard"):
            mui.Typography("Dashboard", variant="h4")
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

elif page == "Chatbot":
    st.header("AI Chatbot")
    st.write("Ask me anything about data science or your uploaded data!")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you today?"]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    # Initialize chatbot
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        temperature=st.session_state.temp,
    )
    chat_model = ChatHuggingFace(llm=llm)
    
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
        memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True
    )

    # Chat interface
    response_container = st.container()
    text_container = st.container()

    with text_container:
        query = st.text_input("Your question:", key="chat_input")
        if query:
            with st.spinner("Thinking..."):
                conversation_string = get_conversation_string()
                refined_query = query_refiner(conversation_string, query)
                st.subheader("Refined Query:")
                st.write(refined_query)
                context = find_match(refined_query)
                response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)

    with response_container:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
