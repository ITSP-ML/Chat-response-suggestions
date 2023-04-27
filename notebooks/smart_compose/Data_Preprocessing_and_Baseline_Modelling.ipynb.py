# %%
"""
Import Libraries & load files
"""

# %%
# import and load data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import re
import pickle
import email
from tqdm import tqdm
import datetime
from dateutil import parser
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense,Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

nltk.download('maxent_ne_chunker')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

# !pip install --upgrade --no-cache-dir gdown
# # !gdown --id 1da9Ci96s3oOPb_ouKTqSbIAoBMvt8hhn

# !pip install bpemb

# from bpemb import BPEmb



# %%
!curl --header "Host: storage.googleapis.com" --header "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36" --header "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header "Accept-Language: en-US,en;q=0.9,ro;q=0.8,hi;q=0.7" --header "Referer: https://www.kaggle.com/" "https://storage.googleapis.com/kaggle-data-sets/8327/11650/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220621%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220621T040855Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=2ce2eec7053c3589bda2345d2db0ad5e3857a82dca76077e6a421089cc51c648155ed4c6684ccd38fce60117b1db78e87790508db90a0c0519399d7f1f98cba1ec7a28fc532660bd489ba962e6442ae146a1ab5faf66f521b4cde9f8c2b0b895c6439d712ef852c95c7d5140a25e5389d17957f087a7f95e69d029fc87cc03b4710003e1d985a0692f255df7f6ea551c7caf3445f3229ad759816278d6e0c359af1fb66d3c807343052783148c8912d30f4b5e39d04f9426d26ad83ae493ab2574cb78d6020905a3ef44b542142d0554da04aacab46eb07e41166cdefa306763341f1a04c6250b800e26b9807bf921ff1604dad453c2922db613bf47e029d3c1" -L -o "archive.zip"

# %%
# !unzip /content/archive.zip
!unzip /content/glove.840B.300d.zip

# %%
"""
### Data Preprocessing    
Reference :- https://github.com/NitishVSawant/Email-Smart-Compose
"""

# %%
"""
Since we have sequence prediction problem we have to figure out what the input and output to the model will be.
"""

# %%
"""

* Load the preprocessed data
* Drop all empty sequences
* Drop all email-sentences with rare words
* Take emails within a word range.      
These strategies will also restrict the size of data
"""

# %%
# load the data
df = pd.read_csv("data/smart_compose/agent_data_cleaned.csv")
df.head()

# %%
# Drop all empty sequences
df = df[df.isna().msg == False]
df = df[df['msg']!='']


# %%
df.shape

# %%
# Drop all emails with rare words
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(df['msg'])

words = list(dict(tokenizer.word_counts).values())
words.sort(reverse=True)

plt.figure(figsize=(15,5))
plt.plot(words)
plt.xlabel('# Words')
plt.ylabel('Counts')
plt.title('Hist-plot of word counts')
plt.ylim(0,200)
plt.show()

# %%
"""
As we can see most words are very rare and useless. Any word which occur more than 100 times will be useful.
"""

# %%
vocab = dict()
for i,j in dict(tokenizer.word_counts).items():
    if j>=25:
        vocab[i]=j

# %%
len(vocab)

# %%
msg_to_keep = []
for i in tqdm(range(df.shape[0])):
    for j in df.msg.iloc[i].lower().split():
        if j not in vocab:
            break
    else:
        msg_to_keep.append(i)

# %%
df = df.iloc[msg_to_keep]
df.reset_index(drop=True,inplace=True)
df.shape

# %%
# take emails within a word range
df['body_wct'] = [len(x.split()) for x in df['msg'].tolist()] 
temp = df['body_wct'].sort_values(ascending=False).reset_index(drop=True).values
plt.plot(temp)
plt.ylim(0,100)
plt.xlabel("# sentences")
plt.ylabel("Length of each sentence(in terms of words)")
plt.show()

# %%
"""
We'll take emails within 30 words and not 70(~ elbow-point) as more than that will explode our data and it'll be difficult to train our data given our resources
"""

# %%
df = df[df.msg_wct<=30]

# %%
df.shape

# %%
# save final data
df.to_csv('data/smart_compose/Preprocessed_agent_data.csv',index=False)

# %%
df.head(10)

# %%
"""
#### Prepare data for Enc-dec, GPT & Transformers 
"""

# %%
# # change the data into input and output sequences for our Encoder-decoder architectures
# def split_sequence_randomly(df):
#     '''
#     To convert our data into encoder-decoder sequences we take random splits between contiguous spans of words.    
#     No. of random splis to take per sentence i.e. k=3
#     '''
#     k = 3 # no.of random splits to take for each body
#     enc_seq = []
#     dec_seq = []
#     for ele in tqdm(df.Body):
#         ele = ele.strip().split(' ')
#         choices = np.random.choice(len(ele),size=k,replace=False)
#         choices = choices[choices>2] # atleast 3 words needed in encoder sequence
#         if len(choices) > 0:
#             for r in choices:
#                 es = ' '.join(ele[:r])
#                 enc_seq.append(es.strip())
#                 ds = ' '.join(ele[r:])
#                 dec_seq.append(ds.strip())
#         else:
#             # if the mail length is very short i.e. ~5 then we're unlikely get splitting index >= 3,therefore we'll manually split at 3
#             r = max(3,np.random.choice(len(ele)))
#             es = ' '.join(ele[:r])
#             enc_seq.append(es.strip())
#             ds = ' '.join(ele[r:])
#             dec_seq.append(ds.strip())

#     # put all in a dataframe
#     data = pd.DataFrame()
#     data['enc_seq'] = enc_seq
#     data['dec_seq'] = dec_seq

#     return data

# %%
# change the data into input and output sequences for our Encoder-decoder architectures
def split_sequence(df):
    '''
    To convert our data into encoder-decoder sequences we split sentences between contiguous spans of words.    
    sentences with < 5 words are dropped
    '''
    enc_seq = []
    dec_seq = []

    for text in tqdm(df.msg):
        sent = text.split()
        for i in range(5,len(sent)):
            enc_seq.append(' '.join(sent[:i]))
            dec_seq.append(' '.join(sent[i:]))

    # put all in a dataframe
    data = pd.DataFrame()
    data['enc_seq'] = enc_seq
    data['dec_seq'] = dec_seq

    return data

# %%
train_data,test_data = train_test_split(df, test_size=0.20, random_state=42)
for ele in [train_data,test_data]:
    ele.reset_index(inplace=True,drop=True)

train_sequences = split_sequence(train_data)
test_sequences = split_sequence(test_data)

# drop empty decoder-sequences
train_sequences = train_sequences[train_sequences.dec_seq!='']
test_sequences = test_sequences[train_sequences.dec_seq!='']

train_data_gpt = train_data[['msg']]
test_data_gpt = test_data[['msg']]

# %%
train_sequences.shape,test_sequences.shape

# %%
train_sequences.head(500)[2:9]

# %%
# save data for Enc-dec models
dataset = [train_sequences,test_sequences]

with open('Sequence_data.pickle', 'wb') as file:
    pickle.dump(dataset, file)

# save data for decoder only models
dataset2 = [train_data_gpt,test_data_gpt]

with open('gpt_data.pickle', 'wb') as file:
    pickle.dump(dataset2, file)


# %%
"""
### ***Start running from here***
"""

# %%
"""
* Add start & end tokens to sequences
* Use pre-trained glove embeddings for words, version :- 2.2M Vocab, trained on 840B tokens.
* Create data pipeline

"""

# %%
with open('Sequence_data.pickle', 'rb') as file:
    train_sequences,test_sequences = pickle.load(file)

# %%
# add start & end tokens
train_sequences['dec_seq_in'] = '<start>' + ' ' + train_sequences['dec_seq']
train_sequences['dec_seq_out'] = train_sequences['dec_seq'] + ' ' + '<end>'

# to avoid having 2 tokenizers for decoder-seq_input & decoder-seq_output, we'll add <end> token to one of the decoder-seq_input sentences.  
train_sequences['dec_seq_in'].iloc[0] = train_sequences['dec_seq_in'].iloc[0] + ' ' + '<end>'
train_sequences.drop('dec_seq',axis=1,inplace=True)


# Similarly for test data

test_sequences['dec_seq_in'] = '<start>' + ' ' + test_sequences['dec_seq']
test_sequences['dec_seq_out'] = test_sequences['dec_seq'] + ' ' + '<end>'
test_sequences.drop('dec_seq',axis=1,inplace=True)


# show data
train_sequences.head()

# %%
# fit tokenizer to enc and dec sequences
Enc_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',oov_token='<unk>')
Enc_tokenizer.fit_on_texts(train_sequences.enc_seq)

Dec_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',oov_token='<unk>')
Dec_tokenizer.fit_on_texts(train_sequences.dec_seq_in)

# %%
# vocab sizes
vocab_size_enc = len(Enc_tokenizer.word_index) + 1
vocab_size_dec = len(Dec_tokenizer.word_index) + 1
print("Encoder vocab size",vocab_size_enc)
print("Decoder vocab size",vocab_size_dec)

print('<start> token index: {} and <end> token index: {} (for decoder-tokenizer)'.format(Dec_tokenizer.word_index['<start>'], Dec_tokenizer.word_index['<end>']))

# %%
tokenizers = [Enc_tokenizer,Dec_tokenizer]
with open('tokenizers.pickle', 'wb') as file:
    pickle.dump(tokenizers, file)

# %%
"""
Run the cell below it
"""

# %%
# create embedding matrix from glove embeddings for enc & dec
embeddings_index = dict()
f = open('glove.840B.300d.txt', encoding="utf8")
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix_enc = np.zeros((vocab_size_enc, 300))
for word, i in Enc_tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_enc[i] = embedding_vector


embedding_matrix_dec = np.zeros((vocab_size_dec, 300))
for word, i in Dec_tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_dec[i] = embedding_vector


# save embeddings for later use
embed_data = [embedding_matrix_enc,embedding_matrix_dec]
with open('embed.pickle', 'wb') as file:
    pickle.dump(embed_data, file)

# %%
# save embeddings
# !gdown --id 1d_vK1OlJnhhEbTgfVgC2OzoIGWhcgGdr
# with open('embed.pickle', 'wb') as file:
#     pickle.dump(embed_data, file)

with open('embed.pickle', 'rb') as file:
    embedding_matrix_enc,embedding_matrix_dec = pickle.load(file)

# %%
print(embedding_matrix_enc.shape,embedding_matrix_dec.shape)

# %%
# This code creates a data generator from scratch

class Dataset:
    def __init__(self, data, Enc_tokenizer, Dec_tokenizer, max_len):
        self.encoder_inps = data['enc_seq'].values
        self.decoder_inps = data['dec_seq_in'].values
        self.decoder_outs = data['dec_seq_out'].values
        self.tknizer_enc = Enc_tokenizer
        self.tknizer_dec = Dec_tokenizer
        self.max_len = max_len

    def __getitem__(self, i):  # magic function which will treat object 'Dataset' as an iterator which can fetch values
        self.encoder_seq = self.tknizer_enc.texts_to_sequences([self.encoder_inps[i]]) # need to pass list of values
        self.decoder_inp_seq = self.tknizer_dec.texts_to_sequences([self.decoder_inps[i]])
        self.decoder_out_seq = self.tknizer_dec.texts_to_sequences([self.decoder_outs[i]])

        self.encoder_seq = pad_sequences(self.encoder_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_inp_seq = pad_sequences(self.decoder_inp_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_out_seq = pad_sequences(self.decoder_out_seq, maxlen=self.max_len, dtype='int32', padding='post')
        return self.encoder_seq, self.decoder_inp_seq, self.decoder_out_seq  # returning padded sequences

    def __len__(self): # your model.fit_gen requires this function
        return len(self.encoder_inps)  #rows in data

    
class Dataloader(tf.keras.utils.Sequence):    
    def __init__(self, dataset, batch_size=1,training=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dataset.encoder_inps))
        self.training = training


    def __getitem__(self, i):
            print(i)
            start = i * self.batch_size
            stop = (i + 1) * self.batch_size
            print(start, stop)
            data = []
            for j in range(start, stop):  # collect all data points in a batch
                data.append(self.dataset[j])

            batch = [np.squeeze(np.stack(samples, axis=1), axis=0) for samples in zip(*data)] 
            # we are creating data like ([enc-seq, dec-inp-seq], dec-out-seq) these are already converted into seq
            return tuple([[batch[0],batch[1]],batch[2]])

    def __len__(self):  # your model.fit_gen requires this function
        if self.training:
            return len(self.indexes) // self.batch_size   # basically return the total no.of batches
        else:
            return len(self.indexes) // self.batch_size   # take a random sample of 100k test data points for evaluation @ every epoch

    def on_epoch_end(self): # shuffling data on every epoch end
        self.indexes = np.random.permutation(self.indexes)
        print(self.indexes.max)

# %%
train_sequences

# %%
MAX_LEN = 30
batch_size = 512

train_dataset = Dataset(train_sequences, Enc_tokenizer, Dec_tokenizer, MAX_LEN)
test_dataset  = Dataset(test_sequences, Enc_tokenizer, Dec_tokenizer, MAX_LEN)

train_dataloader = Dataloader(train_dataset, batch_size,training=True)
test_dataloader = Dataloader(test_dataset, batch_size,training=False)


print(train_dataloader[0][0][0].shape, train_dataloader[0][0][1].shape, train_dataloader[0][1].shape) # enc_inp,dec_inp,dec_out/target

# %%
train_dataloader.dataset

# %%
"""
### Helper functions
"""

# %%
def get_callbacks(filepath,log_dir):
    '''
    To load all required callbacks
    '''
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss',  verbose=1, save_best_only=True,save_weights_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.7,patience=2)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1,restore_best_weights=True)

    def changeLearningRate(epoch,lr):
        print('leaning rate has been changed')
        if epoch > 0 and (epoch+1)%2==0:
            lr = lr*(1-0.2)  
        return lr 

    lrschedule = tf.keras.callbacks.LearningRateScheduler(changeLearningRate, verbose=1)
    Callbacks = [checkpoint,reduce_lr,earlystop,tensorboard_callback,lrschedule]
    return Callbacks

# %%
def initialize_model(path):
    '''
    We need to run the model first then load weights , for a subclassed model
    '''
    
    # link = "https://drive.google.com/drive/folders/1pLjhBcjPph81Qz75c23RhrvAqsNJzvgz?usp=sharing"
    # import gdown
    # gdown.download_folder(link,quiet=True)

    model  = Encoder_Decoder(vocab_size_enc,vocab_size_dec,emb_dim=300,units=256,input_length=MAX_LEN,batch_size=batch_size)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy')  # as targets aren't one hot encoded in data-generator
    model.fit(train_dataloader, steps_per_epoch=1, epochs=1, validation_data=test_dataloader, validation_steps=1)

    model.load_weights(path)
    return model

# %%
"""
### Modelling
"""

# %%
"""
Baseline Vanilla Enc-Dec
"""

# %%
class Encoder(tf.keras.layers.Layer):
    
    def __init__(self,vocab_size,emb_dim,units,input_length):
        super().__init__()
        self.embedding = Embedding(input_dim=vocab_size,output_dim=emb_dim,weights=[embedding_matrix_enc],input_length=input_length,mask_zero=True,trainable=False)
        self.lstm = LSTM(units=units,return_state=True,return_sequences=True,dropout=0.5)
        self.units = units

    def call(self,inps,initial_states):
        enc_input = self.embedding(inps)
        enc_out,enc_h,enc_c = self.lstm(enc_input,initial_state=initial_states)
        return enc_out,enc_h,enc_c

    def initialize_states(self,batch_size):
        state_h = tf.zeros(shape=(batch_size,self.units))
        state_c = tf.zeros(shape=(batch_size,self.units))
        return state_h,state_c

# %%
class Decoder(tf.keras.layers.Layer):

    def __init__(self,vocab_size,emb_dim,units,input_length):
        super().__init__()
        self.embedding = Embedding(input_dim=vocab_size,output_dim=emb_dim,weights=[embedding_matrix_dec],input_length=input_length,mask_zero=True,trainable=False)
        self.lstm = LSTM(units=units,return_state=True,return_sequences=True,dropout=0.5)

    def call(self,inps,enc_h,enc_c):
        dec_input = self.embedding(inps)
        dec_out,dec_h,dec_c = self.lstm(dec_input,initial_state=[enc_h,enc_c])
        return dec_out,dec_h,dec_c


# %%
class Encoder_Decoder(tf.keras.models.Model):

    def __init__(self,vocab_size_enc,vocab_size_dec,emb_dim,units,input_length,batch_size):
        super().__init__()
        # emb_dim,units,input_length --> all are same for encoder and decoder
        self.encoder = Encoder(vocab_size_enc,emb_dim,units,input_length)
        self.decoder = Decoder(vocab_size_dec,emb_dim,units,input_length)
        self.dense = Dense(units=vocab_size_dec,activation='softmax')
        self.batch_size = batch_size

    def call(self,data):
        enc_input,dec_input = data[0],data[1]
        states = self.encoder.initialize_states(self.batch_size)
        enc_out,enc_h,enc_c = self.encoder(enc_input,states)
        dec_out,dec_h,dec_c = self.decoder(dec_input,enc_h,enc_c)

        output = self.dense(dec_out)  # batch x timestep x vocab
        print('output', output.shape)

        return output


# %%
"""
#### Training Part-1
"""

# %%
train_dataloader.__len__() / batch_size


# %%
log_dir = './logs_model1'
filepath = "best_enc_dec1.hdf5"
Callbacks = get_callbacks(filepath,log_dir)

# %%
model  = Encoder_Decoder(vocab_size_enc,vocab_size_dec,emb_dim=300,units=256,input_length=MAX_LEN,batch_size=batch_size)
optimizer = tf.keras.optimizers.Adam(0.005)
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy')  # as targets aren't one hot encoded in data-generator
train_steps = train_dataloader.__len__()
valid_steps = test_dataloader.__len__()


# %%
test_dataloader.indexes

# %%
model.fit(train_dataloader, steps_per_epoch=train_steps, epochs=20, validation_data=test_dataloader, validation_steps=valid_steps,callbacks=Callbacks)

# %%
train_dataloader.__len__()

# %%
44*512

# %%
%reload_ext tensorboard
%tensorboard --logdir logs_model1

# %%
"""
#### Training Part-2
"""

# %%
log_dir = './logs_model2'
filepath = "best_enc_dec2.hdf5"

# %%
optimizer = tf.keras.optimizers.RMSprop(0.0001)
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy')  
train_steps = train_dataloader.__len__()
valid_steps = test_dataloader.__len__()
model.fit(train_dataloader, steps_per_epoch=train_steps, epochs=20, validation_data=test_dataloader, validation_steps=valid_steps,callbacks=Callbacks)

# %%
%reload_ext tensorboard
%tensorboard --logdir logs_model2

# %%
"""
#### Training Part-3
"""

# %%
model = initialize_model('best_enc_dec1.hdf5')

# %%
tf.keras.backend.clear_session()
!rm -rf ./logs_model3
# model = initialize_model()


# %%
log_dir = './logs_model3'
filepath = "best_enc_dec3.hdf5"
Callbacks = get_callbacks(filepath,log_dir)

# %%
optimizer = tf.keras.optimizers.RMSprop(0.00005)
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy')  # as targets aren't one hot encoded in data-generator
train_steps = train_dataloader.__len__()
valid_steps = test_dataloader.__len__()
model.fit(train_dataloader, steps_per_epoch=train_steps, epochs=20, validation_data=test_dataloader, validation_steps=valid_steps,callbacks=Callbacks)

# %%
%reload_ext tensorboard
%tensorboard --logdir logs_model3

# %%
"""
### Inference
"""

# %%
# load the model
model = initialize_model('/best_enc_dec1.hdf5')

# %%
# test_dataloader = Dataloader(test_dataset, batch_size,training=False)
test_steps = test_dataloader.__len__()
model.evaluate(test_dataloader,steps=test_steps)

# %%
reverse_dict = dict(map(reversed, Dec_tokenizer.word_index.items()))  
def predict(input_sentence,l=30):
    text_to_id = Enc_tokenizer.texts_to_sequences([input_sentence])
    inp_seq = np.array(text_to_id).reshape(1,-1)  # batch x ids
    intial_states = model.encoder.initialize_states(batch_size=1)
    _,state_h,state_c = model.encoder(inp_seq,intial_states)
    curr_vec = np.full((1,1), 2, dtype=int) # batch x id for <start> token
    end_token_index = 9185
    p = []
    for _ in range(l):
        dec_out,state_h,state_c = model.decoder(curr_vec,state_h,state_c)
        out = model.dense(dec_out)
        pred_idx = np.argmax(out)
        if pred_idx==end_token_index:
            break
        p.append(reverse_dict.get(pred_idx))
        curr_vec = np.reshape(pred_idx,(1,1))
    return p


# %%
def get_prediction_for_sample(data):
    for _ in range(10):
        idx = np.random.choice(data.shape[0])
        input_sentence = data.iloc[idx].enc_seq
        target_sentence = data.iloc[idx].dec_seq_out.split()[:-1]
        print("Input:",input_sentence)
        print('='*130)
        print("Output:",target_sentence)
        print('='*130)
        p = predict(input_sentence)
        print("Prediction:",p)
        print()

get_prediction_for_sample(train_sequences)
print('-*'*130)
get_prediction_for_sample(test_sequences)

# %%
import nltk.translate.bleu_score as bleu
# bleu score for train
sample = train_sequences.sample(100,ignore_index=True,replace=False,random_state=0)
reference_inp = sample.enc_seq.str.lower().values.tolist()
reference_tar = sample.dec_seq_out.str.lower().values.tolist()
reference_tar = [[ele.strip().split()[:-1]] for ele in reference_tar]  # changing to the format 'corpus_bleu' takes in & removing the <end> token

prediction = []
for sent in tqdm(reference_inp):
    p = predict(sent)
    prediction.append(p)
print('BLEU score for train data: {}'.format(bleu.corpus_bleu(reference_tar, prediction)))

# ,smoothing_function=SmoothingFunction().method2

# bleu score for test
sample = test_sequences.sample(100,ignore_index=True,replace=False,random_state=0)
reference_inp = sample.enc_seq.str.lower().values.tolist()
reference_tar = sample.dec_seq_out.str.lower().values.tolist()
reference_tar = [[ele.strip().split()[:-1]] for ele in reference_tar]  # changing to the format 'corpus_bleu' takes in

prediction = []
for sent in tqdm(reference_inp):
    p = predict(sent)
    prediction.append(p)
    
print('\nBLEU score for test data: {}'.format(bleu.corpus_bleu(reference_tar, prediction)))

# %%
"""
The Baseline benchmark for Categorical Log-loss on a random sample of 100k test-points of test set is **1.071**.   
The Baseline benchmark for BLEU Score for random 100 test-points of test set is **0.10**
"""

# %%
"""
### Latency-check
"""

# %%
!gdown --id 1Kjh_eBcsuVHj5EBFLGCr67I4xTTlU35D

# %%
model = initialize_model('/content/best_enc_dec3.hdf5')

# %%
def get_latency():
    t = []
    sample = test_sequences.sample(100,ignore_index=True,replace=False,random_state=0)
    reference_inp = sample.enc_seq.str.lower().values.tolist()
    for sent in reference_inp:
        a = time.time()
        p = predict(sent)
        b = time.time()
        ms = (b-a) * 1000 # time in milliseconds
        t.append(ms)

    return t

# %%
t = get_latency()

# %%
np.mean(t),np.percentile(t,90),np.percentile(t,99)

# %%
"""
END
"""