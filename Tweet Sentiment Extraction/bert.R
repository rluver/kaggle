require("data.table")
require("dplyr")
require("stringr")
require("keras")




# user function
getPreprocessedData = function(train, test){
  
  encoded_train_text = mapply(tokenizer$encode, max_len = seq_length, train$text)
  encoded_train_selected_text = mapply(tokenizer$encode, max_len = seq_length, train$selected_text)
  encoded_train_sentiment = mapply(tokenizer$encode, max_len = seq_length, train$sentiment)
  
  encoded_test_text = mapply(tokenizer$encode, max_len = seq_length, test$text)
  encoded_test_sentiment = mapply(tokenizer$encode, max_len = seq_length, test$sentiment)
  
  
  encoded_train_text = mapply(as.matrix, encoded_train_text[seq(1, nrow(train) * 2, 2)]) %>% t() %>% list()
  encoded_train_selected_text = mapply(as.matrix, encoded_train_selected_text[seq(1, nrow(train) * 2, 2)]) %>% t() %>% list()
  encoded_train_sentiment = mapply(as.matrix, encoded_train_sentiment[seq(1, nrow(train) * 2, 2)]) %>% t() %>% list()
  
  encoded_test_text = mapply(as.matrix, encoded_test_text[seq(1, nrow(test) * 2, 2)]) %>% t() %>% list()
  encoded_test_sentiment = mapply(as.matrix, encoded_test_sentiment[seq(1, nrow(test) * 2, 2)]) %>% t() %>% list()
  
  return(list(encoded_train_text, 
              encoded_train_selected_text, 
              encoded_train_sentiment, 
              encoded_test_text, 
              encoded_test_sentiment))
}


# data load
c(submission, test, train) %<-% mapply(fread, list.files("D:/tweet-sentiment-extraction", full.names = T),
                                       SIMPLIFY = T, USE.NAMES = F)

# model param
seq_length = 165L
embed_hidden_size = 128
batch_size = 64
epochs = 500
learning_rate = 1e-4
vocab_size = keras_bert$load_vocabulary("D:/uncased_L-12_H-768_A-12/vocab.txt") %>% length() + 1

# load bert
keras_bert = reticulate::import("keras_bert")
tokenizer = keras_bert$load_vocabulary("D:/uncased_L-12_H-768_A-12/vocab.txt") %>% 
  keras_bert$Tokenizer()

model = keras_bert$load_trained_model_from_checkpoint(
  config_file = "D:/uncased_L-12_H-768_A-12/bert_config.json",
  checkpoint_file = "D:/uncased_L-12_H-768_A-12/bert_model.ckpt",
  training = T,
  trainable = T,
  seq_len = seq_length)

# get preprocess
c(train_text, train_selected_text, train_sentiment, 
  test_text, test_sentiment) %<-% getPreprocessedData(train, test)



# determine input, output and concat
sentence = layer_input(shape = c(165)) %>% 
  layer_embedding(input_dim = vocab_size,
                  output_dim = embed_hidden_size) %>% 
  layer_dropout(rate = 0.5)

sentiment = layer_input(shape = c(1)) %>% 
  layer_embedding(input_dim = vocab_size,
                  output_dim = embed_hidden_size) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_lstm(units = embed_hidden_size) %>% 
  layer_repeat_vector(n = 165)

merged = list(sentence, sentiment) %>% 
  layer_add() %>% 
  layer_lstm(units = embed_hidden_size) %>% 
  layer_dropout(rate = 0.5)

pred = merged %>% 
  layer_dense(units = vocab_size,
              activation = 'softmax')


# model
model = keras_model(inputs = list(sentence, sentiment),
                    outputs = pred) %>% 
  # compile
  compile(optimizer = 'Nadam',
          loss = 'categorical_crossentropy',
          metrics = 'accuracy')


# submissionn
