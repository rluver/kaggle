import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold, GroupKFold
from transformers import *
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



# data load
train = pd.read_csv('D:/nlp-getting-started/train.csv')
test = pd.read_csv('D:/nlp-getting-started/test.csv')
submission = pd.read_csv('D:/nlp-getting-started/sample_submission.csv')


tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2', do_lower_case = True)
model_albert = TFAlbertModel.from_pretrained('albert-xlarge-v2')


# parameter
seq_len = 50

def preprocess():
    
    train['tokenized'] = train.text.apply(lambda x: tokenizer.encode(x, max_length = seq_len))
    
    return(sequence.pad_sequences(train.tokenized, maxlen = seq_len, padding = 'post'))
    

def modelBuild():
    
    input = Input((seq_len), dtype = tf.int32, name = 'input_token')    
    layer1, _ = model_albert(input)
    mean_pooling = GlobalAveragePooling1D()(layer1)
    max_pooling = GlobalMaxPooling1D()(layer1)
    pooling = concatenate([mean_pooling, max_pooling])
    drop1 = Dropout(0.5,
                    name = 'drop1')(pooling)
    layer2 = Dense(units = 1024,
                   activation = 'selu',
                   name = 'layer2')(drop1)               
    drop2 = Dropout(0.5,
                    name = 'drop2')(layer2)
    output = Dense(units = 1, 
                   activation = 'sigmoid',
                   kernel_initializer = glorot_normal(),
                   bias_initializer = glorot_normal(),
                   name = 'output')(drop2)

    
    model = Model(input, output)
    print(model.summary())
    model.compile(optimizer = Nadam(learning_rate = 0.0001),
                  metrics = 'accuracy',
                  loss = 'binary_crossentropy')
    
    return model


train_tokenized = preprocess()

model = modelBuild()

model.fit(train_tokenized,
          np.array(train.target),
          batch_size = 4,
          epochs = 3,
          validation_split = 0.3)