import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

st.title("ChatGPT-like clone with LLama3")

# Initialize the model and prompt template
model = OllamaLLM(model="llama3")
template = """
Here is the conversation history: {context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = ""

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and model response
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context = st.session_state.context + "\n" + " ".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages]
        )
        response = chain.invoke({"context": context, "question": prompt})
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.context = context + f"\nAssistant: {response}"