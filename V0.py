import streamlit as st
import openai  # Correct import

st.title("Angel's Awesome Chatbot")

openai.api_key = st.secrets["API_key"]  # Correct API key initialization

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Say something")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        stream = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
            stream=True
        )
        response_text = ""
        for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                response_text += chunk["choices"][0]["delta"].get("content", "")

            st.write(response_text)  # Display the response

    st.session_state.messages.append({"role": "assistant", "content": response_text})  # Store response correctly
