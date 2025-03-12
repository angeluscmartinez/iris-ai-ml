import streamlit as st
import openai

st.title("Angel's Awesome Chatbot")

openai.api_key = st.secrets["API_key"]

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    role = "👤 User" if message["role"] == "user" else "🤖 Assistant"
    st.markdown(f"**{role}:** {message['content']}")

# Use st.text_input() instead of chat_input for better compatibility
prompt = st.text_input("Say something", "")

if prompt:
    st.markdown(f"**👤 User:** {prompt}")  # Display user input
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )

    response_text = response["choices"][0]["message"]["content"]
    st.markdown(f"**🤖 Assistant:** {response_text}")  # Display assistant response

    # Save to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})


