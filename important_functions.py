import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone with new API
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ai-assistant"

# Ensure index exists
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of 'all-MiniLM-L6-v2' embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
index = pc.Index(index_name)

def find_match(input):
    try:
        input_em = model.encode(input).tolist()
        result = index.query(vector=input_em, top_k=2, include_metadata=True)
        if len(result['matches']) == 0:
            return "No matches found."
        text_output = ""
        for match in result['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                text_output += match['metadata']['text'] + "\n"
        return text_output.strip()
    except Exception as e:
        return f"Error querying Pinecone: {str(e)}"

@st.cache_resource
def load_gpt2_model():
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def query_refiner(conversation, query):
    tokenizer, model = load_gpt2_model()
    prompt = f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=256, num_return_sequences=1, no_repeat_ngram_size=2)
    refined_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    refined_query = refined_query.split("Refined Query:")[-1].strip()
    return refined_query

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string
