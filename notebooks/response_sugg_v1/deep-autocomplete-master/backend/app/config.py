import json
import torch
import os

# load vocabulary
vocab_path = os.path.join("app", "saved_models", "vocab", "char2int_2.json")
with open(vocab_path, "r") as f:
    char2int = json.load(f)


int2char = {v: k for k, v in char2int.items()}

# define parameters
VOCAB_SIZE = len(int2char)
HIDDEN_SIZE = 512
N_LAYERS = 3
P_DROPOUT = 0.4
BATCH_FIRST = True
# PATH = os.path.join("app", "saved_models", "charRNN_questions_epoch_20.pt")
PATH = "C:/Users/feress/Documents/myprojects/Chat-response-suggestions/notebooks/response_sugg_v1/deep-autocomplete-master/backend/app/saved_models/char_rnn_epoch_20.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
