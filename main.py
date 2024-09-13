import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import requests
import json

st.set_page_config(page_title="Faithful AI")
st.title(r"$\textsf{\Large Faithful AI}$ 🤖✝️")
st.markdown("Your Christian chatbot to have conversations regarding the Bible :)")
st.markdown("---")

# Initialize the model and prompt template
model = OllamaLLM(model="llama3")
template = "Here is the conversation history: {context}\n\nQuestion: {question}\n\nAnswer:"
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Call the Ollama API to generate a response
def call_ollama_api(prompt): 
    try:
        response = requests.post(
            st.secrets["OLLAMA_API_URL"], 
            headers={"Content-Type": "application/json"}, 
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        data = json.loads(response.text)
        return data["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API: {e}")
        return "Error: Unable to connect to Ollama API"


# Handle conversation using the Ollama API
@st.cache_data(show_spinner=False)
def handle_conversation(context, user_input):
    prompt = f"Here is the conversation history: {context}\n\nQuestion: {user_input}\n\nAnswer:"
    return call_ollama_api(prompt)


# Handle user input and generate a response
def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        with st.spinner("Loading..."):
            result = handle_conversation(st.session_state.context, user_input)
        st.session_state.context += f"\nYou: {user_input}\nAI: {result}"
        st.session_state.chat_dialogue.append(f"You: {user_input}")
        st.session_state.chat_dialogue.append(f"AI: {result}")
        st.session_state.user_input = ""


# Initialize the Streamlit app
def initialize_streamlit():
    if "chat_dialogue" not in st.session_state:
        st.session_state.chat_dialogue = []
    if "context" not in st.session_state:
        st.session_state.context = """
        Human: You are a Christian chatbot that can answer questions and have conversations regarding the Bible. 
        You are a friendly chatbot that is here to help people learn more about the Bible and Christianity.
        If you do not know the answer to a question, you can simply say that you don't know.
        Every answer that you give, please refer to the Bible as the source of truth, and provide the chapter and verse if possible.
        Show enthusiasm and be positive in your responses. You are also able to speak in Spanish, French, Italian, and German.
        """
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Display conversation history
    for message in st.session_state.chat_dialogue:
        with st.chat_message("assistant" if message.startswith("AI:") else "user"):
            st.markdown(message)

    # Get user input and handle the conversation
    st.text_input(label="You", placeholder="Message Faithful AI", key="user_input", on_change=handle_input)
    
    # Position the input text at the bottom
    style = """
        <style>
        .stTextInput {
            bottom: 3rem;
            position: fixed;
            padding: 1rem;
        }
        .stApp {
            margin-bottom: 5rem;
        }
        </style>
        """
    st.markdown(style, unsafe_allow_html=True)


if __name__ == "__main__":
    initialize_streamlit()
    