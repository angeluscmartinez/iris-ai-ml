import streamlit as st
import openai  # ✅ Correct import

st.title("Angel's Awesome Chatbot")

# Set OpenAI API key properly
openai.api_key = st.secrets["API_key"]  # ✅ Correct way

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
    response = openai.ChatCompletion.create(  # ✅ Correct method
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )

    response_text = response["choices"][0]["message"]["content"]  # ✅ Extract response
    st.markdown(f"**🤖 Assistant:** {response_text}")  # Display assistant response

    # Save to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})







