import os
os.environ['TF_KERAS'] = '1'

import numpy as np
import pandas as pd
from transformers import *
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Model

# load data
train = pd.read_csv('d:/nlp-getting-started/train.csv')
test = pd.read_csv('d:/nlp-getting-started/test.csv')
submission = pd.read_csv('d:/nlp-getting-started/sample_submission.csv')

# set parameter
MAX_LEN = 70
EPOCH = 3
BATCH_SIZE = 16

# load albert
tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2', do_lower_case = True)
albert = TFAlbertModel.from_pretrained('albert-xlarge-v2')


# user define function
def encode(self, tokenizer, max_len = 100):
       
    text = self.text.apply(lambda x: tokenizer.tokenize(x))
    text = text.apply(lambda x: x[:68])
    text = text.apply(lambda x: ['[CLS]'] + x + ['[SEP]'])
    text = text.apply(lambda x: tokenizer.convert_tokens_to_ids(x))
    
    token = sequence.pad_sequences(text, maxlen = MAX_LEN, padding = 'post', dtype = 'int32')
    length = text.apply(lambda x: len(x))
    mask = list(map(lambda x: [1] * x + [0] * (MAX_LEN - x), length))
    seg_id = [[0] * MAX_LEN] * self.shape[0]
       
    return np.array(token), np.array(mask), np.array(seg_id)


def build_model(model, max_len = 100, num_labels = 1):
    
    input_word = Input(shape = (max_len, ), dtype = tf.int32, name = 'input_word')
    input_mask = Input(shape = (max_len, ), dtype = tf.int32, name = 'input_mask')
    input_seg_id = Input(shape = (max_len, ), dtype = tf.int32, name = 'input_seg_id')

    input_ = [input_word, input_mask, input_seg_id]        
    _, temp = model([input_word, input_mask, input_seg_id])    
    temp1 = Dense(units = 512,
                  name = 'temp_layer1')(temp)
    output_ = Dense(units = num_labels,
                    name = 'output')(temp1)
    
    model = Model(inputs = input_,
                  outputs = output_)
    model.compile(optimizer =Nadam(lr = 3e-5),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    return model



# preprocess
train_ = encode(train, tokenizer, max_len = MAX_LEN)
test_ = encode(test, tokenizer, max_len = MAX_LEN)

# model
model = build_model(model = albert,
                    max_len = MAX_LEN,
                    num_labels = 1)

# train
history = model.fit(
    train_,
    np.array(train.target),
    validation_split = 0.3,
    epochs = EPOCH,
    batch_size = BATCH_SIZE)
