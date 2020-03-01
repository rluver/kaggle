require("data.table")
require("dplyr")
require("stringr")
require("tm")
require("text2vec")
require("glmnet")
require("reticulate")



# data load
# abbreviation
# abbreviations : https://www.kaggle.com/rftexas/text-only-kfold-bert 
# contractions : http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python 
word_change = fread("d:/dic.csv", header = T) %>% 
  bind_rows(fread("d:/contractions.csv", header = T))

tweet = fread("D:/nlp-getting-started/train.csv", header = T, encoding = "UTF-8") %>% 
  mutate(text = str_to_lower(text) %>% 
           str_replace_all("(?<=https?)://t.co/[\\w]{1,10}", "") %>% 
           #str_replace_all(word_change$from, word_change$to) %>% 
           str_replace_all("[^A-z0-9]", " ") %>% 
           str_replace_all("\\b\\B", "") %>% 
           str_replace_all("[\\[\\]_]", ""))





# split data
ind = sample(2, nrow(tweet), replace = T, prob = c(0.8, 0.2))

train = tweet %>% filter(ind == 1) %>% select(text) %>% unlist() %>% as.vector() %>% str_replace_all("\\W", " ")
label_train = tweet %>% filter(ind == 1) %>% select(target) %>% unlist() %>% as.vector()

valid = tweet %>% filter(ind == 2) %>% select(text) %>% unlist() %>% as.vector() %>% str_replace_all("\\W", " ")
label_valid = tweet %>% filter(ind == 2) %>% select(target) %>% unlist() %>% as.vector()




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

mutate(text = str_to_lower(text) %>% 
         str_replace_all("(?<=https?)://t.co/[\\w]{1,10}", "") %>% 
         #str_replace_all(word_change$from, word_change$to) %>% 
         str_replace_all("[^A-z0-9]", " ") %>% 
         str_replace_all("\\b\\B", "") %>% 
         str_replace_all("[\\[\\]_]", ""))



# test
test = fread("D:/nlp-getting-started/test.csv", header = T) %>% 
  select(text) %>% unlist() %>% as.vector() %>% str_to_lower() %>% 
  str_replace_all("(?<=https?)://t.co/[\\w]{1,10}", "") %>% 
  str_replace_all("[^A-z0-9]", " ") %>% 
  str_replace_all("\\b\\B", "") %>% 
  str_replace_all("[\\[\\]_]", "")

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
fwrite(submission, "d:/submission_text2vec_preprocess.csv", row.names = F)
