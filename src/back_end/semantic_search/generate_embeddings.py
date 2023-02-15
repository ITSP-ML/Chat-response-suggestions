import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

from src.back_end.preprocess import get_agent_msgs
def generate_embeddings(model_name, data_path, save_path):
    """
    model_name: the name of model used for generating embeddings
    data_path: data_path for data to be encoded
    """
    # load model
    model = SentenceTransformer(model_name)
    # read data
    data = get_agent_msgs(data_path)
    # generate embeddings
    encoded_data = model.encode(data.processed_msg.tolist())
    encoded_data = np.asarray(encoded_data.astype('float32'))
    # save embeddings
    with open(save_path, 'wb') as pkl:
        pickle.dump(encoded_data, pkl)
    return encoded_data
