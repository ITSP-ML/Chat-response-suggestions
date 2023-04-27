# %%
import nltk
nltk.download('punkt')
import string
from transformers import AutoTokenizer

# %%
model_checkpoint = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# %%
# load dataseet 
import pandas as pd 
data = pd.read_csv('data/smart_compose/Preprocessed_agent_data.csv')
data.head()

# %%
# change the data into input and output sequences for our Encoder-decoder architectures
from tqdm import tqdm
def split_sequence(df):
    '''
    To convert our data into encoder-decoder sequences we split sentences between contiguous spans of words.    
    sentences with < 5 words are dropped
    '''
    enc_seq = []
    dec_seq = []

    for text in tqdm(df.msg):
        sent = text.split()
        for i in range(3,len(sent)):
            x = ' '.join(sent[:i])
            y = ' '.join(sent[i:])
            x = 'sos '+ x+ ' eos'
            y = 'sos '+ y+ ' eos'
            enc_seq.append(x)
            dec_seq.append(y)

    # put all in a dataframe
    data = pd.DataFrame()
    data['enc_seq'] = enc_seq
    data['dec_seq'] = dec_seq

    return data

# %%

data = split_sequence(data)
data.head()

# %%
data.shape

# %%
df = data
df.columns = ["input", "output"]

# %%
import numpy as np
from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['input']),np.array(df['output']),test_size=0.1,random_state=0,shuffle=True)

# %%
_, a, _,_ = train_test_split(np.array(df['input']),np.array(df['output']),test_size=0.1,random_state=0,shuffle=True)
a

# %%
y_tr[3]

# %%
from keras.preprocessing.text import Tokenizer 
from keras_preprocessing.sequence import pad_sequences

# %%
#this function returns max length of largest sentence  
def max_length(t):
  return max(len(i) for i in t)

#prepare a tokenizer for Input on training data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_tr))

# %%
#this counts how many words we have in input sentence my input vocab size
tot_cnt = 0
for key,value in x_tokenizer.word_counts.items():
    tot_cnt=tot_cnt+1

# %%
#convert text sequences into integer sequences
x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)
max_inp_len = max_length(x_tr_seq)
#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_inp_len, padding='post')
x_val   =   pad_sequences(x_val_seq, maxlen=max_inp_len, padding='post')

#size of vocabulary (for padding token)
x_voc   = tot_cnt + 1

# %%
#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_tr))

# %%
tot_cnt = 0
for key,value in y_tokenizer.word_counts.items():
    tot_cnt=tot_cnt+1

# %%
#convert text sequences into integer sequences
y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_val)
max_out_len = max_length(y_tr_seq)
#padding zero upto maximum length
y_tr    =   pad_sequences(y_tr_seq,  maxlen=max_out_len, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=max_out_len, padding='post')

#size of vocabulary ( +1 for padding token)
y_voc   = tot_cnt + 1

# %%
#this is the manual implementation of AdditiveAttention
import tensorflow as tf
import os
from tensorflow.python.keras import backend as K


class AttentionLayer(tf.keras.layers.Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

# %%
#https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
from keras import backend as K
import tensorflow as tf
K.clear_session()


latent_dim = 300
embedding_dim= 300
#https://github.com/thushv89/attention_keras/blob/master/examples/nmt/model.py
# Encoder
encoder_inputs = tf.keras.layers.Input(shape=(max_inp_len,))

#embedding layer
enc_emb =  tf.keras.layers.Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)

#encoder lstm 1
encoder_lstm = tf.keras.layers.LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output, state_h, state_c = encoder_lstm(enc_emb)
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = tf.keras.layers.Input(shape=(None,))

#embedding layer
dec_emb_layer = tf.keras.layers.Embedding(y_voc, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

# Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_output, decoder_outputs])

# Concat attention input and decoder LSTM output
decoder_concat_input = tf.keras.layers.Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#dense layer
#https://datascience.stackexchange.com/questions/10836/the-difference-between-dense-and-timedistributeddense-of-keras
decoder_dense =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# %%
# Define the model 

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])


# %%
def fit(model):
  # we use 20% of our data for validation.
  history = model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:],
                      validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]),
                      batch_size= 128,
                      epochs= 10)
  return history

# %%
def plot_loss(history):
  # Plot the results of the training.
  import matplotlib.pyplot as plt
  plt.plot(history.history['loss'], label="Training loss")
  plt.plot(history.history['val_loss'], label="Validation loss")
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.show()

# %%
def plot_acuraccy(history):
  # Plot the results of the training.
  import matplotlib.pyplot as plt
  plt.plot(history.history['sparse_categorical_accuracy'], label="Training accuraccy")
  plt.plot(history.history['val_sparse_categorical_accuracy'], label="Validation accuraccy")
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Accuraccy')
  plt.show()

# %%
his= fit(model)


# %%
plot_loss(his)


# %%
plot_acuraccy(his)

# %%
# Encode the input sequence to get the feature vector
encoder_model = tf.keras.Model(inputs=encoder_inputs,outputs=[encoder_output, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = tf.keras.layers.Input(shape=(latent_dim,))
decoder_state_input_c = tf.keras.layers.Input(shape=(latent_dim,))
decoder_hidden_state_input = tf.keras.layers.Input(shape=(max_inp_len,latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = tf.keras.layers.Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat) 

# Final decoder model
decoder_model = tf.keras.Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

# %%

encoder_model.save('my_last_model.h5')
# encoder_model_file = drive.CreateFile({'title' : 'my_last_model.h5'})                       
# encoder_model_file.SetContentFile('my_last_model.h5')                       
# encoder_model_file.Upload()

# %%
decoder_model.save('my_last_decoder_model.h5')
# decoder_model_file = drive.CreateFile({'title' : 'my_last_decoder_model.h5'})                       
# decoder_model_file.SetContentFile('my_last_decoder_model.h5')                       
# decoder_model_file.Upload()

# %%
import pickle
with open('train.pickle', 'wb') as f:
    pickle.dump([x_tr, y_tr ,x_val,y_val,x_tokenizer,y_tokenizer,max_inp_len,max_out_len], f)

# %%
uploaded = drive.CreateFile({'title': 'train.pickle'})
uploaded.SetContentFile('train.pickle')
uploaded.Upload()

# %%
from nltk.translate.bleu_score import sentence_bleu
def calculate_bleu(original,tokenize_sentence):
  references = [original]
  candidates = tokenize_sentence
  score = sentence_bleu(references, candidates)
  return score

# %%
reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index
input_word_index= x_tokenizer.word_index

# %%
#THIS function predicts the next sentence
def bring_my_sentence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    print(e_out, e_h, e_c)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sos']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eos'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eos'  or len(decoded_sentence.split()) >= (max_out_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

# %%
bring_my_sentence(np.array(['hi there how']).reshape(1,1))

# %%
#converting tensor vector to sentence
def seq2output(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sos']) and i!=target_word_index['eos']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString
#converting input tensor to sentence
def seq2input(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=input_word_index['sos']) and i!=input_word_index['eos']):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

# %%
input = 'can i'
input_seq   =   x_tokenizer.texts_to_sequences({input})
print(input_seq)
x   =   pad_sequences(input_seq, maxlen=max_inp_len, padding='post')
print(x)
bring_my_sentence(x.reshape(1,max_inp_len))

# %%
max_inp_len

# %%
output = []
from nltk import word_tokenize
for i in range(40,50):
  pred = bring_my_sentence(x_val[i].reshape(1,max_inp_len))
  score = calculate_bleu(word_tokenize(seq2output(y_val[i])),word_tokenize(pred))
  print('----------------------------------------------------------')
  print('input seq :',seq2input(x_val[i]))
  print("Predicted seq :",pred)
  print("BLEU SCORE: ",score)
  print("----------------------------------------------------------")
  output.append({"Input seq":seq2input(x_val[i]), "Pred. Seq":pred ,"BLEU SCORE": score})
results_df = pd.DataFrame.from_dict(output) 

# %%
pd.set_option('display.max_colwidth', None)
results_df 

# %%
data[data.input.str.contains('I will gladly check on it for you')]

# %%
data

# %%
