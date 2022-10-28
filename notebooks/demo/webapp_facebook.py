# importing libraries
import streamlit as st
import requests
import json
import requests

# get_text is a simple function to get user input from text_input


def get_text_customer(msg):
    input_text = st.text_input("Customer: ", msg)
    return input_text


st.sidebar.title("ITSP Bot")

st.title(
    """
ITSP Bot
ITSP Bot is an NLP conversational AI
"""
)


customer_input = get_text_customer("hello")


if st.button("send"):
    res = requests.post(
        "http://localhost:8004/",
        headers={
            #'User-agent'  : 'Internet Explorer/2.0',
            "Content-type": "application/json"
        },
        json={"text": customer_input},
    )
    st.text_area(
        "conversation till now:",
        value=res.json()["generated_responses"][-1],
        height=500,
        max_chars=None,
        key=1,
    )

try:
    res = requests.post(
    "http://localhost:8004/default",
    headers={
        #'User-agent'  : 'Internet Explorer/2.0',
        "Content-type": "application/json"
    },
    json={"text": "this message will not processed"},
)
    st.text_area(
    "conversation till now:",
    value=res.json()["generated_responses"][-1],
    height=500,
    max_chars=None,
    key=1,
)
except:
    print('this should be called only once per convertation')
