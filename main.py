import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model and prompt template
model = OllamaLLM(model="llama3")
template = """
Here is the conversation history: {context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Function to handle the conversation
@st.cache_resource(show_spinner=False)
def handle_conversation(context, user_input):
    try:
        result = chain.invoke({"context": context, "question": user_input})
    except ConnectionError as e:
        # Log the error (assuming a logging mechanism is in place)
        print(f"Connection error: {e}")
        # Return a user-friendly message
        return "There was a connection error. Please try again later."
    return result

# Function to deal with Streamlit app
def initialize_streamlit():
    st.set_page_config(page_title="Faithful AI")

    # Set up the left sidebar
    st.title(r"$\textsf{\Large Faithful AI}$ 🤖✝️")
    st.markdown("Your Christian chatbot to have conversations regarding the Bible :)")
    st.markdown("---")

    # Initialize the session state variables
    if "chat_dialogue" not in st.session_state:
        st.session_state["chat_dialogue"] = []
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.5
    if "top_p" not in st.session_state:
        st.session_state["top_p"] = 0.9
    if "max_seq_len" not in st.session_state:
        st.session_state["max_seq_len"] = 500
    if "context" not in st.session_state:
        st.session_state.context = """
        Human: You are a Christian chatbot that can answer questions and have conversations regarding the Bible. 
        You are a friendly chatbot that is here to help people learn more about the Bible and Christianity.
        If you do not know the answer to a question, you can simply say that you don't know.
        Every answer that you give, please refer to the Bible as the source of truth, and provide the chapter and verse if possible.
        Show enthusiasm and be positive in your responses. You are also able to speak in Spanish, French, Italian, German, and Chinese.
        """

    # Get user input and handle the conversation
    user_input = st.text_input(label="You", placeholder="Message Faithful AI")
    if user_input or st.button("Send", ):
        with st.spinner("Loading..."):
            result = handle_conversation(st.session_state.context, user_input)
        
        # Display user input
        with st.chat_message("user"):
            st.markdown(f"You: {user_input}")

        # Store the conversation history and context    
        st.session_state.context += f"\nYou: {user_input}\nAI: {result}"
        st.session_state.chat_dialogue.append(f"You: {user_input}")
        st.session_state.chat_dialogue.append(f"AI: {result}")

    # Display AI response
    for message in st.session_state.chat_dialogue:
        if message.startswith("AI:"):
            with st.chat_message("assistant"):
                st.markdown(message)

if __name__ == "__main__":
    initialize_streamlit()