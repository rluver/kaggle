import pandas as pd
import ktrain
from ktrain import text


# data load
train = pd.read_csv('D:/nlp-getting-started/train.csv')
test = pd.read_csv('D:/nlp-getting-started/test.csv')
submission = pd.read_csv('D:/nlp-getting-started/sample_submission.csv')


model_albert = text.Transformer('albert-xlarge-v2', maxlen = 128, class_names = list(set(train.target)))
model_albert_train = model_albert.preprocess_train(list(train.text), list(train.target))
model_albert_classifier = model_albert.get_classifier()
lr = ktrain.get_learner(model_albert_classifier,
                        train_data = model_albert_train,
                        use_multiprocessing = True,
                        multigpu = True,
                        batch_size = 64)
lr.fit_onecycle(5e-5, 5)
