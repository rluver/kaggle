# refer to https://www.kaggle.com/mobassir/understanding-cross-lingual-models


import numpy as np
import pandas as pd
import re
import torch
from transformers import *


# data load
PATH = 'D:/jigsaw-multilingual-toxic-comment-classification/'
train1 = pd.read_csv(PATH + 'jigsaw-toxic-comment-train.csv')
train2 = pd.read_csv(PATH + 'jigsaw-unintended-bias-train.csv')
train2.toxic = train2.toxic.round().astype(int)
valid = pd.read_csv(PATH + 'validation.csv')
test = pd.read_csv(PATH + 'test.csv')
submission = pd.read_csv(PATH + 'sample_submission.csv')

# join
train = pd.concat(
    [train1[['comment_text', 'toxic']],
     train2[['comment_text', 'toxic']]]
    )


# parameter
MAX_LEN = 192


# user define function
# cleansing
def cleansing(self):
    # punctuation
    punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

    # abbreviation
    abbre = {"aren't" : "are not",
             "can't" : "cannot",
             "couldn't" : "could not",
             "couldnt" : "could not",
             "didn't" : "did not",
             "doesn't" : "does not",
             "doesnt" : "does not",
             "don't" : "do not",
             "hadn't" : "had not",
             "hasn't" : "has not",
             "haven't" : "have not",
             "havent" : "have not",
             "he'd" : "he would",
             "he'll" : "he will",
             "he's" : "he is",
             "i'd" : "I would",
             "i'd" : "I had",
             "i'll" : "I will",
             "i'm" : "I am",
             "isn't" : "is not",
             "it's" : "it is",
             "it'll":"it will",
             "i've" : "I have",
             "let's" : "let us",
             "mightn't" : "might not",
             "mustn't" : "must not",
             "shan't" : "shall not",
             "she'd" : "she would",
             "she'll" : "she will",
             "she's" : "she is",
             "shouldn't" : "should not",
             "shouldnt" : "should not",
             "that's" : "that is",
             "thats" : "that is",
             "there's" : "there is",
             "theres" : "there is",
             "they'd" : "they would",
             "they'll" : "they will",
             "they're" : "they are",
             "theyre":  "they are",
             "they've" : "they have",
             "we'd" : "we would",
             "we're" : "we are",
             "weren't" : "were not",
             "we've" : "we have",
             "what'll" : "what will",
             "what're" : "what are",
             "what's" : "what is",
             "what've" : "what have",
             "where's" : "where is",
             "who'd" : "who would",
             "who'll" : "who will",
             "who're" : "who are",
             "who's" : "who is",
             "who've" : "who have",
             "won't" : "will not",
             "wouldn't" : "would not",
             "you'd" : "you would",
             "you'll" : "you will",
             "you're" : "you are",
             "you've" : "you have",
             "'re": " are",
             "wasn't": "was not",
             "we'll":" will",
             "didn't": "did not",
             "tryin'":"trying"}
       
    for abb in abbre.items():
        self = re.sub(abb[0], abb[1], self)
        
    for x in punct:
        self = self.replace(x, '')
    
    self = self.lower()
    self = re.sub('\\n', ' ', self)
    self = re.sub('\[\[User.*', '', self)
    self = re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', self)
    self = re.sub('\(http://.*?\s\(http://.*\)', '', self)
    
    return self


# encode
def encoding(text, tokenizer, maxlen = MAX_LEN):
    enc_dic = tokenizer.batch_encode_plus(
        text,
        return_attention_masks = False,
        return_token_type_ids = False,
        pad_to_max_length = True,
        max_length = maxlen
    )
    
    return np.array(enc_dic['input_ids'])


# cleansing
train.comment_text = train.comment_text.apply(lambda x: cleansing(x))
valid.comment_text = valid.comment_text.apply(lambda x: cleansing(x))
test.content = test.content.apply(lambda x: cleansing(x))

x_train = encoding(train.comment_text.astype(str).values, tokenizer, maxlen = MAX_LEN)
x_valid = encoding(valid.comment_text.astype(str).values, tokenizer, maxlen = MAX_LEN)
x_test = encoding(test.content.astype(str).values, tokenizer, maxlen = MAX_LEN)


# model
MODEL = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(MODEL)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AlbertForMaskedLM.from_pretrained(MODEL)

# build model
def build_model(model, max_len = MAX_LEN):
    input_word_ids = Input(shape = (max_len, ), dtype = tf.int32, name = 'input_word_ids')
    sequence_output = model(input_word_ids)[0]
    class_token = sequence_output[:, 0, :]
    layer = Dropout(rate = 0.3)(class_token)
    output = Dense(units = 1,
                   activation = 'sigmoid')(layer)
    
    model = Model(inputs = input_word_ids,
                  outputs = output)
    model.compile(Nadam(lr = 3e-5), loss = 'binary_crossentropy', metrics = [AUC()])
    
    return model
