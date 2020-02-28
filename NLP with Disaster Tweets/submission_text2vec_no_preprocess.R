require("data.table")
require("dplyr")
require("stringr")
require("text2vec")
require("glmnet")




# data load
text = fread("D:/nlp-getting-started/train.csv", header = T)




# split data
ind = sample(2, nrow(text), replace = T, prob = c(0.8, 0.2))

train = text %>% filter(ind == 1) %>% select(text) %>% unlist() %>% as.vector() %>% str_replace_all("\\W", " ")
label_train = text %>% filter(ind == 1) %>% select(target) %>% unlist() %>% as.vector()

valid = text %>% filter(ind == 2) %>% select(text) %>% unlist() %>% as.vector() %>% str_replace_all("\\W", " ")
label_valid = text %>% filter(ind == 2) %>% select(target) %>% unlist() %>% as.vector()




# tokenization
i_train = itoken(train,
                 str_to_lower,
                 tokenizer = word_tokenizer,
                 progressbar = T)

i_valid = itoken(valid,
                str_to_lower,
                tokenizer = word_tokenizer,
                progressbar = T)

# create vocab vector
vectorizer = create_vocabulary(i_train) %>% 
  vocab_vectorizer()

# create tfidf matrix
train_tfidf = create_dtm(i_train, vectorizer) %>% 
  fit_transform(TfIdf$new())

valid_tfidf = create_dtm(i_valid, vectorizer) %>% 
  fit_transform(TfIdf$new())




# training
classifier_glmnet = cv.glmnet(x = train_tfidf,
                              y = label_train,
                              family = "binomial",
                              type.measure = "auc",
                              trace.it = T)




# result
# plot
plot(classifier_glmnet)

# confusion matrix
caret::confusionMatrix(classifier_glmnet %>% predict(valid_tfidf, type = "class") %>% as.factor(),
                       label_valid %>% as.factor())




# test
test = fread("D:/nlp-getting-started/test.csv", header = T) %>% 
  select(text) %>% unlist() %>% as.vector() %>% str_replace_all("\\W", " ")

i_test = itoken(test,
                str_to_lower,
                tokenizer = word_tokenizer,
                progressbar = T)

test_tfidf = create_dtm(i_test, vectorizer) %>% 
  fit_transform(TfIdf$new())



# submission
submission = fread("D:/nlp-getting-started/sample_submission.csv", header = T) %>% 
  mutate(target = classifier_glmnet %>% predict(test_tfidf, type = "class") %>% as.numeric())




# write
fwrite(submission, "d:/submission_text2vec_no_preprocess.csv", row.names = F)