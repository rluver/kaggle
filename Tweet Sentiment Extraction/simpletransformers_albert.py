import pandas as pd
import numpy as np
import json
from simpletransformers.question_answering import QuestionAnsweringModel


# data load
train = pd.read_csv('d:/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('d:/tweet-sentiment-extraction/test.csv')
submission = pd.read_csv('d:/tweet-sentiment-extraction/sample_submission.csv')

train = np.array(train)
test = np.array(test)


# user define function
# convert train
def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1

def do_qa_train(train):

    output = []
    for line in train:
        context = line[1]

        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(answer) != str or type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answer_starts = find_all(context, answer)
        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer.lower()})
            break
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        output.append({'context': context.lower(), 'qas': qas})
        
    return output

qa_train = do_qa_train(train)

   
# convert test
def do_qa_test(test):
    output = []
    for line in test:
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        if type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answers = []
        answers.append({'answer_start': 1000000, 'text': '__None__'})
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
        output.append({'context': context.lower(), 'qas': qas})
    return output

qa_test = do_qa_test(test)



# model
model = QuestionAnsweringModel('albert',
                               'albert-large-v2',
                               args = {'reprocess_input_data': True,
                                     'overwrite_output_dir': True,
                                     'learning_rate': 3e-5,
                                     'num_train_epochs': 3,
                                     'max_seq_length': 192,
                                     'doc_stride': 64,
                                     'fp16': False,
                                     },
                               use_cuda = True)

# train
model.train_model(qa_train)


# predict
pred = model.predict(qa_test)
pred = pd.DataFrame.from_dict(pred)

submission.selected_text = pred.answer
submission.to_csv('submission.csv', index = False)