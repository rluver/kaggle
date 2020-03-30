require("data.table")
require("dplyr")
require("keras")




# data load
data = fread('d:/digit-recognizer/train.csv', header = T)
test = fread('d:/digit-recognizer/test.csv', header = T)
submission = fread('d:/digit-recognizer/sample_submission.csv', header = T)



# data split
c(x_train, y_train) %<-% list((data %>% select(-1))/255, data %>% select(1))
x_test = test/255

x_train = x_train %>% 
  as.matrix() %>% 
  array(dim = c(nrow(.), 28, 28)) %>% 
  array_reshape(dim = c(nrow(.), 28, 28, 1))
    
x_test = x_test %>% 
  as.matrix() %>% 
  array(dim = c(nrow(.), 28, 28)) %>% 
  array_reshape(dim = c(nrow(.), 28, 28, 1))

y_train = y_train$label %>% to_categorical()




# model
# define activation
mish = function(x){
  x*activation_tanh(activation_softplus(x)) %>% return()
}

# cnn
model_cnn = keras_model_sequential() %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(4, 4),
                input_shape = c(28, 28, 1),
                activation = mish) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(4, 4),
                activation = mish) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(2, 2),
                activation = mish) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(2, 2),
                activation = mish) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512,
              activation = mish) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10,
              activation = 'softmax') %>% 
  # compile
  compile(
    loss = loss_categorical_crossentropy,
    optimizer = optimizer_nadam(),
    metrics = 'accuracy'
  )

# train
history = model_cnn %>% 
  fit(
    x_train,
    y_train,
    epochs = 100,
    validation_split = 0.3,
    callbacks = list(
      callback_early_stopping(monitor = 'val_loss',
                              mode = 'min',
                              verbose = 1,
                              patience = 7),
      callback_reduce_lr_on_plateau(monitor = 'val_loss',
                                    factor = 0.5,
                                    patience = 3,
                                    verbose = 1,
                                    min_lr = 0.000001)
    )
  )




# save
submission$Label = model_cnn %>% predict(x_test) %>% max.col() - 1
fwrite(submission, 'd:/submission_cnn_mish.csv', row.names = F)
