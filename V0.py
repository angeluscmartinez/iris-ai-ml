import streamlit as st
from openai import OpenAI  # Correct import for OpenAI SDK v1.66.3+

st.title("Angel's Awesome Chatbot")

# Initialize OpenAI client correctly
client = OpenAI(api_key=st.secrets["API_key"])  # New correct API initialization

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    role = "👤 User" if message["role"] == "user" else "🤖 Assistant"
    st.markdown(f"**{role}:** {message['content']}")

# User input
prompt = st.text_input("Say something", "")

if prompt:
    st.markdown(f"**👤 User:** {prompt}")  # Display user input
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Correct API call for OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )

    response_text = response.choices[0].message.content  # Extract correct response
    st.markdown(f"**🤖 Assistant:** {response_text}")  # Display assistant response

    # Save to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})




