# chatbot.py
# Import necessary modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import streamlit as st
import httpx

# Define a prompt template for the chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the questions"),
        ("user", "Question:{question}")
    ]
)

# Set up the Streamlit framework
st.title('Langchain Chatbot With LLAMA3 model')  # Set the title of the Streamlit app
input_text = st.text_input("Ask your question!")  # Create a text input field in the Streamlit app

# Test connectivity to the Llama3 model endpoint
try:
    response = httpx.get("https://api.ollama.ai/v1/models/llama3")  # Replace with actual endpoint
    if response.status_code == 200:
        st.success("Successfully connected to Llama3 model endpoint.")
    else:
        st.error(f"Failed to connect to Llama3 model endpoint. Status code: {response.status_code}")
except httpx.ConnectError as e:
    st.error(f"Connection error: {e}")

# Initialize the Ollama model
llm = OllamaLLM(model="llama3")

# Create a chain that combines the prompt and the Ollama model
chain = prompt | llm

# Invoke the chain with the input text and display the output
if input_text:
    try:
        response = chain.invoke({"question": input_text})
        st.write(response)
    except httpx.ConnectError as e:
        st.error(f"Connection error: {e}")