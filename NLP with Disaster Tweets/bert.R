require("data.table")
require("dplyr")
require("stringr")
require("reticulate")
require("keras")




# set sys.env
Sys.setenv(TF_KERAS = 1)




# user function
# tokenize text
tokenize_fun_train = function(dataset) {
  c(indices, target, segments) %<-% list(list(),list(),list())
  for ( i in 1:nrow(dataset)) {
    c(indices_tok, segments_tok) %<-% tokenizer$encode(dataset[[DATA_COLUMN]][i], 
                                                       max_len=seq_length)
    indices = indices %>% append(list(as.matrix(indices_tok)))
    target = target %>% append(dataset[[LABEL_COLUMN]][i])
    segments = segments %>% append(list(as.matrix(segments_tok)))
  }
  return(list(indices,segments, target))
}

tokenize_fun_test = function(dataset) {
  c(indices, segments) %<-% list(list(),list())
  for ( i in 1:nrow(dataset)) {
    c(indices_tok, segments_tok) %<-% tokenizer$encode(dataset[[DATA_COLUMN]][i], 
                                                       max_len = seq_length)
    indices = indices %>% append(list(as.matrix(indices_tok)))
    segments = segments %>% append(list(as.matrix(segments_tok)))
  }
  
  return(list(indices, segments))
}

# read data
dt_data_train = function(dir){
  data = data.table::fread(dir)
  c(x_train, x_segment, y_train) %<-% tokenize_fun(data)
  return(list(x_train, x_segment, y_train))
}  

dt_data_test = function(dir){
  data = data.table::fread(dir)
  c(x_test, x_segment_test) %<-% tokenize_fun_test(data)
 
  return(list(x_test, x_segment_test))
}
  



# model param
seq_length = 50L
batch_size = 70
epochs = 2
learning_rate = 1e-4

# user param
c(DATA_COLUMN, LABEL_COLUMN) %<-% c("text", "target")




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
c(x_train, x_segment, y_train) %<-% dt_data("D:/nlp-getting-started/train.csv")
c(x_test, x_segment_test) %<-% dt_data_test("D:/nlp-getting-started/test.csv")
submission = fread("D:/nlp-getting-started/sample_submission.csv", header = T)



# cal decay and warmup step
c(decay_steps, warmup_steps) %<-% keras_bert$calc_train_steps(do.call(cbind, y_train) %>% t() %>% length(),
                                                              batch_size = batch_size,
                                                              epochs = epochs)




# determine input, output and concat
model = keras_model(inputs = list(input1 = get_layer(model, name = "Input-Token")$input, 
                                  input2 = get_layer(model, name = "Input-Segment")$input), 
                    outputs = get_layer(model, name = "NSP-Dense")$output %>% 
                      layer_dense(units = 1L,
                                  activation = "sigmoid",
                                  kernel_initializer = initializer_truncated_normal(stddev = 0.02),
                                  name = "output"))


# compile and train
model %>% compile(keras_bert$AdamWarmup(decay_steps = decay_steps,
                                        warmup_steps = warmup_steps,
                                        lr = learning_rate),
                  loss = "binary_crossentropy",
                  metrics = "accuracy")

model %>% fit(c(do.call(cbind, x_train) %>% t() %>% list(), 
                do.call(cbind, x_segment) %>% t() %>% list()),
              target,
              epochs = epochs,
              batch_size = batch_size,
              validation_split = 0.3)




# submission
submission$target = ifelse(model %>% predict(c(do.call(cbind, x_test) %>% t() %>% list(),
                                               do.call(cbind, x_segment_test) %>% t() %>% list())) < 0.5,
                           0, 1)
fwrite(submission, 'd:/bert.csv', row.names = F)
