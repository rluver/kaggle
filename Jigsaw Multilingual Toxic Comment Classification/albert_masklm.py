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



# model
MODEL = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(MODEL)
