import pandas as pd
import re
import numpy as np



data = pd.read_csv('data_dump/response_sugg/GODEL/data_v1.csv')
def get_data (chat_id, last_id):
    row_chat = data[data.chat_id == chat_id].iloc[:last_id]
    return row_chat



