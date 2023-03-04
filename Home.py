import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import json
from streamlit_lottie import st_lottie
import creds

openai.api_key = creds.api_key

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

st.set_page_config(page_title="AskiRIS", page_icon=":guardsman:")
image = st.image("Picture1.png", use_column_width=False)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    header = st.title("Welcome to iRIS Sage Artificial Intelligence")

with col2:
    lottie_coding = load_lottiefile("coding.json")
    st_lottie(
        lottie_coding,
        speed=2,
        reverse=False,
        loop=True,
        quality="high",
        height=200,
        width=200,
        key=None,
    )
st.set_option('deprecation.showPyplotGlobalUse', False)

verif_df = pd.read_csv("eis.csv")
req_df = pd.read_csv("requirements.csv")

def generate_response(prompt):
    #Operate on the requirements with "Count the" or "List the"as first characters
    if prompt.startswith("Count the") and len(prompt.split()) >= 2:
        filtered_df = req_df[req_df['Description'].str.contains(prompt.split()[2])]
        return filtered_df.shape[0] if not filtered_df.empty else "No requirements found"        
    elif prompt.startswith("List the") and len(prompt.split()) >= 2:
        if len(prompt.split()) > 2 and prompt.split()[2] == "total":
            return req_df
        else:
            filtered_df = req_df[req_df['Description'].str.contains(prompt.split()[2])]
            return filtered_df if not filtered_df.empty else "No requirements found"
    if prompt.startswith("Count requirements in the") and len(prompt.split()) >= 2:
        filtered_df = req_df[req_df['Title'].str.contains(prompt.split()[4], case=False)]
        return filtered_df.shape[0] if not filtered_df.empty else "No requirements found" 
    elif prompt.startswith("List requirements in the") and len(prompt.split()) >= 2:
        if len(prompt.split()) > 2 and prompt.split()[4] == "total":
            return req_df
        else:
            filtered_df = req_df[req_df['Title'].str.contains(prompt.split()[4], case=False)]
            return filtered_df if not filtered_df.empty else "No requirements found"
    elif "Plot the" in prompt:
        x = verif_df["X"]
        y = verif_df["Y"]
        plt.plot(x, y)
        plt.xlabel("Time [Days]")
        plt.ylabel("Requirements")
        plt.title("Guidance Section Requirements Verification Status")
        st.pyplot()
        return ""  
    elif "Play a cool video" in prompt or "iris" in prompt:
        video = open("Overview.mp4", "rb").read()
        st.video(video, start_time=0)
        return ""
    else:
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=4000,
            n=1,
            stop=None,
            temperature=0.1,
        )
        response = completions.choices[0].text
        print("No matches found, generating response from OpenAI API")
        return response

generated_responses = []

def get_text():
    input_text = st.text_input("You: ", value="", key="user_input")
    return input_text

user_input = get_text()

if user_input:
    response = generate_response(user_input)
    print("Response: ", response)
    st.write("iRIS AI:")
    if isinstance(response, int):
        st.write(f"There are {response} requirements.")
    else:
        st.write(response)