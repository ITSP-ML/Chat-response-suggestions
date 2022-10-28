# importing libraries
import streamlit as st
import requests

# from chatterbot import ChatBot
# from chatterbot.trainers import ListTrainer
# from chatterbot.trainers import ChatterBotCorpusTrainer
import json

# get_text is a simple function to get user input from text_input
def get_text(msg):
    input_text = st.text_input("You: ", msg)
    return input_text


# data input
# data = json.loads(open(r'C:\Users\Jojo\Desktop\projects\chatbot\chatbot\chatbot\data_tolokers.json','r').read())#change path accordingly
# data2 = json.loads(open(r'C:\Users\Jojo\Desktop\projects\chatbot\chatbot\chatbot\sw.json','r').read())#change path accordingly
# tra = []
# for k, row in enumerate(data):
#     #print(k)
#     tra.append(row['dialog'][0]['text'])
# for k, row in enumerate(data2):
#     #print(k)
#     tra.append(row['dialog'][0]['text'])
st.sidebar.title("ITSP Bot")
st.title(
    """
ITSP Bot
ITSP Bot is a retreval base bot
"""
)
# bot = ChatBot(name = 'PyBot', read_only = False,preprocessors=['chatterbot.preprocessors.clean_whitespace','chatterbot.preprocessors.convert_to_ascii','chatterbot.preprocessors.unescape_html'], logic_adapters = ['chatterbot.logic.MathematicalEvaluation','chatterbot.logic.BestMatch'])
# ind = 1
# if st.sidebar.button('Initialize bot'):
#     trainer2 = ListTrainer(bot)
#     trainer2.train(tra)
#     st.title("Your bot is ready to talk to you")
#     ind = ind +1


def get_response(msg):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = "{" + f'"sender":"Me","message": "{msg}"' + "}"
    response = requests.post(
        "http://localhost:5005/webhooks/rest/webhook", headers=headers, data=data
    )
    ## parse the response to get the response text
    result = ""
    for x in response.json():
        result += x["text"] + "\n"
    return result


user_input = get_text("hello")
if True:
    st.text_area(
        "Bot:", value=get_response(user_input), height=200, max_chars=None, key=None
    )
else:
    st.text_area(
        "Bot:",
        value="Please start the bot by clicking sidebar button",
        height=200,
        max_chars=None,
        key=None,
    )
