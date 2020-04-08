require("data.table")
require("dplyr")
require("stringr")
require("reticulate")
require("keras")




# set sys.env
Sys.setenv(TF_KERAS = 1)




# user function
# tokenize text
getPreprocessedData = function(train, test){
  
  encoded_train = mapply(tokenizer$encode, max_len = seq_length, train$text)
  encoded_test = mapply(tokenizer$encode, max_len = seq_length, test$text)
  
  c(x_train, x_segment) %<-% list(mapply(as.matrix, encoded_train[seq(1, nrow(train) * 2, 2)]) %>% t() %>% list(),
                                  mapply(as.matrix, encoded_train[seq(2, nrow(train) * 2, 2)]) %>% t() %>% list())
  y_train = train$target
  
  c(x_test, x_segment_test) %<-% list(mapply(as.matrix, encoded_test[seq(1, nrow(test) * 2, 2)]) %>% t() %>% list(),
                                      mapply(as.matrix, encoded_test[seq(2, nrow(test) * 2, 2)]) %>% t() %>% list())
  
  return(list(x_train, x_segment, y_train, x_test, x_segment_test))
}
  



# model param
seq_length = 50L
batch_size = 70
epochs = 2
learning_rate = 1e-4




# load bert
keras_bert = import("keras_bert")
tokenizer = keras_bert$load_vocabulary("D:/uncased_L-12_H-768_A-12/vocab.txt") %>% 
  keras_bert$Tokenizer()

model = keras_bert$load_trained_model_from_checkpoint(
  config_file = "D:/uncased_L-12_H-768_A-12/bert_config.json",
  checkpoint_file = "D:/uncased_L-12_H-768_A-12/bert_model.ckpt",
  training = T,
  trainable = T,
  seq_len = seq_length)




# load data
c(submission, test, train) %<-% mapply(fread, list.files("D:/nlp-getting-started", full.names = T),
                                       SIMPLIFY = T, USE.NAMES = F)

# get preprocessd data
c(x_train, x_segment, y_train, x_test, x_segment_test) %<-% getPreprocessedData(train, test)



# cal decay and warmup step
c(decay_steps, warmup_steps) %<-% keras_bert$calc_train_steps(y_train %>% length(),
                                                              batch_size = batch_size,
                                                              epochs = epochs)




# determine input, output and concat
model = keras_model(inputs = list(input1 = get_layer(model, name = "Input-Token")$input, 
                                  input2 = get_layer(model, name = "Input-Segment")$input), 
                    outputs = get_layer(model, name = "NSP-Dense")$output %>% 
                      ayer_batch_normalization() %>% 
                      layer_dropout(0.4) %>% 
                      layer_dense(units = 128,
                                  activation = 'selu') %>% 
                      layer_batch_normalization() %>% 
                      layer_dropout(0.4) %>% 
                      layer_dense(units = 1L,
                                  activation = "sigmoid",
                                  kernel_initializer = initializer_truncated_normal(stddev = 0.02),
                                  name = "output")) %>% 
  # compile
  compile(keras_bert$AdamWarmup(decay_steps = decay_steps,
                                warmup_steps = warmup_steps,
                                lr = learning_rate),
          loss = "binary_crossentropy",
          metrics = "accuracy")

# train
model %>% fit(c(x_train, x_segment),
              y_train,
              epochs = epochs,
              batch_size = batch_size,
              validation_split = 0.3)




# submission
submission$target = ifelse(model %>% predict(c(x_test, x_segment_test)) < 0.5,
                           0, 1)
fwrite(submission, 'd:/bert.csv', row.names = F)
