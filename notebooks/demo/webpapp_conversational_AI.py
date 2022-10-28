# importing libraries
import streamlit as st
import requests
import json
import requests

with open("notebooks/demo/convertation.txt") as f:
    lines = f.readlines()
all_chat = "".join(lines)

# get_text is a simple function to get user input from text_input


def get_text_customer(msg):
    input_text = st.text_input("Customer: ", msg)
    return input_text


def get_text_agent(msg):
    input_text = st.text_input("Agent: ", msg)
    return input_text


st.sidebar.title("ITSP Bot")
st.title(
    """
ITSP Bot
ITSP Bot is an NLP conversational AI
"""
)


customer_input = "customer: " + get_text_customer("hello")
agent_input = "agent: " + get_text_agent("")
all_chat += customer_input + "\n"


def get_sugg(all_chat):
    all_chat += agent_input
    res = requests.post(
        "http://localhost:8000/",
        headers={
            #'User-agent'  : 'Internet Explorer/2.0',
            "Content-type": "application/json"
        },
        json={"text": all_chat},
    )
    suggestions = res.json()
    ## parse the response to get the response text
    l = len(all_chat)
    result = ""
    for text in suggestions:
        result += text["generated_text"][l:] + "\n"

    # for x in response.json():
    #     result += x["text"] + "\n"
    return result


if st.button("what should i say !!"):
    suggs = get_sugg(all_chat)
    st.text_area(
        "Sugg:",
        value=suggs,
        height=200,
        max_chars=None,
        key=None,
    )
else:
    st.text_area(
        "Sugg:",
        value="Please press the button to get some suggestions",
        height=200,
        max_chars=None,
        key=None,
    )

if st.button("send"):
    # add new rows
    with open("notebooks/demo/convertation.txt", "a") as f:
        f.write(customer_input)
        f.write("\n")
        f.write(agent_input)
        f.write("\n")
    # read conversation from file
    all_chat += agent_input + "\n"
    st.text_area(
        "conversation till now:",
        value=all_chat,
        height=500,
        max_chars=None,
        key=None,
    )
