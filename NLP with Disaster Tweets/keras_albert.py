import os
os.environ['TF_KERAS'] = '1'


import pandas as pd
import numpy as np
import tensorflow as tf
from keras_albert_model import *
from keras import *
from transformers import *
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import *
from tensorflow.keras.initializers import *
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import *


# data load
train = pd.read_csv('D:/nlp-getting-started/train.csv')
test = pd.read_csv('D:/nlp-getting-started/test.csv')
submission = pd.read_csv('D:/nlp-getting-started/sample_submission.csv')

# load tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2', do_lower_case = True)

# user define function
def preprocess(self):
    
    self['tokenized'] = self.text.apply(lambda x: tokenizer.encode(x, max_length = seq_len))
    
    return (sequence.pad_sequences(self.tokenized, maxlen = seq_len, padding = 'post'))

def segPreprocess(self):
       
    return (np.array([0]*self.shape[0]*seq_len).reshape(self.shape[0],seq_len))
    

# parameter
seq_len = 50
epoch = 3
batch_size = 10

# tokenized
train_tokenized, test_tokenized = [preprocess(train), preprocess(test)]
train_seg, test_seg = [segPreprocess(train), segPreprocess(test)]



# albert
model = build_albert(token_num = 30000,
                     hidden_dim = 2048,
                     head_num = 32,
                     feed_forward_dim = 8192,
                     seq_len = seq_len,
                     training = True)
model.summary()

inputs = model.inputs
sop_dense = model.layers[-3].output
drop1 = Dropout(0.5,
                name = 'dropout1')(sop_dense)
layer2 = Dense(units = 512,
               name = 'layer2')(drop1)
drop2 = Dropout(0.5,
                name = 'dropout2')(layer2)
layer3 = Dense(units = 64,
               name = 'layer3')(drop2)
drop3 = Dropout(0.3,
                name = 'dropout3')(layer3)
output = Dense(units = 1,
               activation = 'sigmoid',
               kernel_initializer = glorot_normal(),
               bias_initializer = glorot_normal(),
               name = 'output')(drop3)



albert_model = Model(inputs = model.inputs,
                     outputs = output)
albert_model.compile(loss = 'binary_crossentropy',
                     optimizer = 'Nadam',
                     metrics = ['accuracy'])
albert_model.fit([train_tokenized, train_seg],
                 np.array(train.target),
                 epochs = epoch,
                 batch_size = batch_size,
                 validation_split = 0.3,
                 shuffle = True)

