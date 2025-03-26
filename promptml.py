import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from streamlit_chat import message
import plotly.express as px
from streamlit_elements import elements, mui, html, dashboard

# Load environment variables
load_dotenv(find_dotenv())

# Set Streamlit page config
st.set_page_config(page_title="PromptML: NextGen AI Assistant", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Settings & Navigation")
    page = st.selectbox("Choose a Page", ["Exploratory Data Analysis (EDA)", "Model Selection", "Prediction", "Chatbot"])
    temperature = st.slider("Temperature (LLM)", 0.01, 1.0, 0.1, 0.01)
    uploaded_files = st.file_uploader("Upload CSV Files", type=["csv", "xlsx"], accept_multiple_files=True)

# Load Data
if uploaded_files:
    dfs = [pd.read_csv(file, low_memory=False) if file.name.endswith(".csv") else pd.read_excel(file) for file in uploaded_files]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
else:
    df = None

# LLM Setup
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={'temperature': temperature},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

if page == "Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")
    if df is not None:
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        tab1, tab2, tab3 = st.tabs(["Overview", "EDA Steps", "Custom Analysis"])
        
        with tab1:
            st.write(df.head())
            st.write(df.describe())
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

elif page == "Model Selection":
    st.header("📌 Model Selection")
    if df is not None:
        model_recommendation = llm("Recommend the best machine learning model for this dataset")
        st.write(model_recommendation)
    else:
        st.warning("Please upload a dataset to get model recommendations.")

elif page == "Prediction":
    st.header("🔮 Make Predictions")
    st.write("Upload test data to get real-time predictions using trained models.")
    if df is not None:
        user_query = st.text_input("Describe your prediction task:")
        if user_query:
            prediction = llm(f"Based on the dataset, make predictions for: {user_query}")
            st.write(prediction)
    else:
        st.warning("Please upload a dataset first.")

elif page == "Chatbot":
    st.header("💬 AI Chatbot")
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
