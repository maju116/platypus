Download images and masks: [2018 Data Science
Bowl](https://www.kaggle.com/c/data-science-bowl-2018).

Build `U-Net` model and compile it with correct loss and metric:

``` r
library(tidyverse)
library(platypus)
library(abind)
library(here)

train_DCB2018_path <- here("development/data-science-bowl-2018/stage1_train")
test_DCB2018_path <- here("development/data-science-bowl-2018/stage1_test")

blocks <- 4 # Number of U-Net convolutional blocks
classes <- 2 # Number of classes
net_h <- 256 # Must be in a form of 2^N
net_w <- 256 # Must be in a form of 2^N

DCB2018_u_net <- u_net(
  input_shape = c(net_h, net_w, 3),
  blocks = blocks,
  classes = classes,
  filters = 16,
  dropout = 0.1,
  batch_normalization = TRUE,
  kernel_initializer = "he_normal"
)

DCB2018_u_net %>%
  compile(
    optimizer = optimizer_adam(lr = 1e-5),
    loss = loss_dice(),
    metrics = metric_dice_coef()
  )

DCB2018_u_net
```

    ## Model
    ## Model: "model"
    ## ________________________________________________________________________________
    ## Layer (type)              Output Shape      Param #  Connected to               
    ## ================================================================================
    ## input_img (InputLayer)    [(None, 256, 256, 0                                   
    ## ________________________________________________________________________________
    ## conv2d (Conv2D)           (None, 256, 256,  448      input_img[0][0]            
    ## ________________________________________________________________________________
    ## batch_normalization (Batc (None, 256, 256,  64       conv2d[0][0]               
    ## ________________________________________________________________________________
    ## re_lu (ReLU)              (None, 256, 256,  0        batch_normalization[0][0]  
    ## ________________________________________________________________________________
    ## conv2d_1 (Conv2D)         (None, 256, 256,  2320     re_lu[0][0]                
    ## ________________________________________________________________________________
    ## batch_normalization_1 (Ba (None, 256, 256,  64       conv2d_1[0][0]             
    ## ________________________________________________________________________________
    ## re_lu_1 (ReLU)            (None, 256, 256,  0        batch_normalization_1[0][0]
    ## ________________________________________________________________________________
    ## max_pooling2d (MaxPooling (None, 128, 128,  0        re_lu_1[0][0]              
    ## ________________________________________________________________________________
    ## dropout (Dropout)         (None, 128, 128,  0        max_pooling2d[0][0]        
    ## ________________________________________________________________________________
    ## conv2d_2 (Conv2D)         (None, 128, 128,  4640     dropout[0][0]              
    ## ________________________________________________________________________________
    ## batch_normalization_2 (Ba (None, 128, 128,  128      conv2d_2[0][0]             
    ## ________________________________________________________________________________
    ## re_lu_2 (ReLU)            (None, 128, 128,  0        batch_normalization_2[0][0]
    ## ________________________________________________________________________________
    ## conv2d_3 (Conv2D)         (None, 128, 128,  9248     re_lu_2[0][0]              
    ## ________________________________________________________________________________
    ## batch_normalization_3 (Ba (None, 128, 128,  128      conv2d_3[0][0]             
    ## ________________________________________________________________________________
    ## re_lu_3 (ReLU)            (None, 128, 128,  0        batch_normalization_3[0][0]
    ## ________________________________________________________________________________
    ## max_pooling2d_1 (MaxPooli (None, 64, 64, 32 0        re_lu_3[0][0]              
    ## ________________________________________________________________________________
    ## dropout_1 (Dropout)       (None, 64, 64, 32 0        max_pooling2d_1[0][0]      
    ## ________________________________________________________________________________
    ## conv2d_4 (Conv2D)         (None, 64, 64, 64 18496    dropout_1[0][0]            
    ## ________________________________________________________________________________
    ## batch_normalization_4 (Ba (None, 64, 64, 64 256      conv2d_4[0][0]             
    ## ________________________________________________________________________________
    ## re_lu_4 (ReLU)            (None, 64, 64, 64 0        batch_normalization_4[0][0]
    ## ________________________________________________________________________________
    ## conv2d_5 (Conv2D)         (None, 64, 64, 64 36928    re_lu_4[0][0]              
    ## ________________________________________________________________________________
    ## batch_normalization_5 (Ba (None, 64, 64, 64 256      conv2d_5[0][0]             
    ## ________________________________________________________________________________
    ## re_lu_5 (ReLU)            (None, 64, 64, 64 0        batch_normalization_5[0][0]
    ## ________________________________________________________________________________
    ## max_pooling2d_2 (MaxPooli (None, 32, 32, 64 0        re_lu_5[0][0]              
    ## ________________________________________________________________________________
    ## dropout_2 (Dropout)       (None, 32, 32, 64 0        max_pooling2d_2[0][0]      
    ## ________________________________________________________________________________
    ## conv2d_6 (Conv2D)         (None, 32, 32, 12 73856    dropout_2[0][0]            
    ## ________________________________________________________________________________
    ## batch_normalization_6 (Ba (None, 32, 32, 12 512      conv2d_6[0][0]             
    ## ________________________________________________________________________________
    ## re_lu_6 (ReLU)            (None, 32, 32, 12 0        batch_normalization_6[0][0]
    ## ________________________________________________________________________________
    ## conv2d_7 (Conv2D)         (None, 32, 32, 12 147584   re_lu_6[0][0]              
    ## ________________________________________________________________________________
    ## batch_normalization_7 (Ba (None, 32, 32, 12 512      conv2d_7[0][0]             
    ## ________________________________________________________________________________
    ## re_lu_7 (ReLU)            (None, 32, 32, 12 0        batch_normalization_7[0][0]
    ## ________________________________________________________________________________
    ## max_pooling2d_3 (MaxPooli (None, 16, 16, 12 0        re_lu_7[0][0]              
    ## ________________________________________________________________________________
    ## dropout_3 (Dropout)       (None, 16, 16, 12 0        max_pooling2d_3[0][0]      
    ## ________________________________________________________________________________
    ## conv2d_8 (Conv2D)         (None, 16, 16, 25 295168   dropout_3[0][0]            
    ## ________________________________________________________________________________
    ## batch_normalization_8 (Ba (None, 16, 16, 25 1024     conv2d_8[0][0]             
    ## ________________________________________________________________________________
    ## re_lu_8 (ReLU)            (None, 16, 16, 25 0        batch_normalization_8[0][0]
    ## ________________________________________________________________________________
    ## conv2d_9 (Conv2D)         (None, 16, 16, 25 590080   re_lu_8[0][0]              
    ## ________________________________________________________________________________
    ## batch_normalization_9 (Ba (None, 16, 16, 25 1024     conv2d_9[0][0]             
    ## ________________________________________________________________________________
    ## re_lu_9 (ReLU)            (None, 16, 16, 25 0        batch_normalization_9[0][0]
    ## ________________________________________________________________________________
    ## conv2d_transpose (Conv2DT (None, 32, 32, 12 295040   re_lu_9[0][0]              
    ## ________________________________________________________________________________
    ## concatenate (Concatenate) (None, 32, 32, 25 0        conv2d_transpose[0][0]     
    ##                                                      re_lu_7[0][0]              
    ## ________________________________________________________________________________
    ## dropout_4 (Dropout)       (None, 32, 32, 25 0        concatenate[0][0]          
    ## ________________________________________________________________________________
    ## conv2d_10 (Conv2D)        (None, 32, 32, 12 295040   dropout_4[0][0]            
    ## ________________________________________________________________________________
    ## batch_normalization_10 (B (None, 32, 32, 12 512      conv2d_10[0][0]            
    ## ________________________________________________________________________________
    ## re_lu_10 (ReLU)           (None, 32, 32, 12 0        batch_normalization_10[0][0
    ## ________________________________________________________________________________
    ## conv2d_11 (Conv2D)        (None, 32, 32, 12 147584   re_lu_10[0][0]             
    ## ________________________________________________________________________________
    ## batch_normalization_11 (B (None, 32, 32, 12 512      conv2d_11[0][0]            
    ## ________________________________________________________________________________
    ## re_lu_11 (ReLU)           (None, 32, 32, 12 0        batch_normalization_11[0][0
    ## ________________________________________________________________________________
    ## conv2d_transpose_1 (Conv2 (None, 64, 64, 64 73792    re_lu_11[0][0]             
    ## ________________________________________________________________________________
    ## concatenate_1 (Concatenat (None, 64, 64, 12 0        conv2d_transpose_1[0][0]   
    ##                                                      re_lu_5[0][0]              
    ## ________________________________________________________________________________
    ## dropout_5 (Dropout)       (None, 64, 64, 12 0        concatenate_1[0][0]        
    ## ________________________________________________________________________________
    ## conv2d_12 (Conv2D)        (None, 64, 64, 64 73792    dropout_5[0][0]            
    ## ________________________________________________________________________________
    ## batch_normalization_12 (B (None, 64, 64, 64 256      conv2d_12[0][0]            
    ## ________________________________________________________________________________
    ## re_lu_12 (ReLU)           (None, 64, 64, 64 0        batch_normalization_12[0][0
    ## ________________________________________________________________________________
    ## conv2d_13 (Conv2D)        (None, 64, 64, 64 36928    re_lu_12[0][0]             
    ## ________________________________________________________________________________
    ## batch_normalization_13 (B (None, 64, 64, 64 256      conv2d_13[0][0]            
    ## ________________________________________________________________________________
    ## re_lu_13 (ReLU)           (None, 64, 64, 64 0        batch_normalization_13[0][0
    ## ________________________________________________________________________________
    ## conv2d_transpose_2 (Conv2 (None, 128, 128,  18464    re_lu_13[0][0]             
    ## ________________________________________________________________________________
    ## concatenate_2 (Concatenat (None, 128, 128,  0        conv2d_transpose_2[0][0]   
    ##                                                      re_lu_3[0][0]              
    ## ________________________________________________________________________________
    ## dropout_6 (Dropout)       (None, 128, 128,  0        concatenate_2[0][0]        
    ## ________________________________________________________________________________
    ## conv2d_14 (Conv2D)        (None, 128, 128,  18464    dropout_6[0][0]            
    ## ________________________________________________________________________________
    ## batch_normalization_14 (B (None, 128, 128,  128      conv2d_14[0][0]            
    ## ________________________________________________________________________________
    ## re_lu_14 (ReLU)           (None, 128, 128,  0        batch_normalization_14[0][0
    ## ________________________________________________________________________________
    ## conv2d_15 (Conv2D)        (None, 128, 128,  9248     re_lu_14[0][0]             
    ## ________________________________________________________________________________
    ## batch_normalization_15 (B (None, 128, 128,  128      conv2d_15[0][0]            
    ## ________________________________________________________________________________
    ## re_lu_15 (ReLU)           (None, 128, 128,  0        batch_normalization_15[0][0
    ## ________________________________________________________________________________
    ## conv2d_transpose_3 (Conv2 (None, 256, 256,  4624     re_lu_15[0][0]             
    ## ________________________________________________________________________________
    ## concatenate_3 (Concatenat (None, 256, 256,  0        conv2d_transpose_3[0][0]   
    ##                                                      re_lu_1[0][0]              
    ## ________________________________________________________________________________
    ## dropout_7 (Dropout)       (None, 256, 256,  0        concatenate_3[0][0]        
    ## ________________________________________________________________________________
    ## conv2d_16 (Conv2D)        (None, 256, 256,  4624     dropout_7[0][0]            
    ## ________________________________________________________________________________
    ## batch_normalization_16 (B (None, 256, 256,  64       conv2d_16[0][0]            
    ## ________________________________________________________________________________
    ## re_lu_16 (ReLU)           (None, 256, 256,  0        batch_normalization_16[0][0
    ## ________________________________________________________________________________
    ## conv2d_17 (Conv2D)        (None, 256, 256,  2320     re_lu_16[0][0]             
    ## ________________________________________________________________________________
    ## batch_normalization_17 (B (None, 256, 256,  64       conv2d_17[0][0]            
    ## ________________________________________________________________________________
    ## re_lu_17 (ReLU)           (None, 256, 256,  0        batch_normalization_17[0][0
    ## ________________________________________________________________________________
    ## conv2d_18 (Conv2D)        (None, 256, 256,  34       re_lu_17[0][0]             
    ## ================================================================================
    ## Total params: 2,164,610
    ## Trainable params: 2,161,666
    ## Non-trainable params: 2,944
    ## ________________________________________________________________________________

Create data generator:

``` r
train_DCB2018_generator <- segmentation_generator(
  path = train_DCB2018_path, # directory with images and masks
  mode = "nested_dirs", # Each image with masks in separate folder
  classes = classes,
  only_images = FALSE,
  target_size = c(net_h, net_w),
  grayscale = FALSE,
  scale = 1 / 255,
  batch_size = 32,
  shuffle = TRUE,
  subdirs = c("images", "masks") # Names of subdirs with images and masks
)
```

    ## [1] "670 images with corresponding masks detected!"

Fit the model (starting from `tensorflow >= 2.1` fitting custom `R`
generators dosen’t work. Please see
[issue](https://github.com/rstudio/keras/issues/1090) and
[issue](https://github.com/rstudio/keras/issues/1073)):

``` r
# DCB2018_u_net %>%
#   fit_generator(
#     train_DCB2018_generator,
#     epochs = 100,
#     steps_per_epoch = 21,
#     callbacks = list(callback_model_checkpoint("development/data-science-bowl-2018/DSB2018_w.hdf5",
#                                                save_best_only = TRUE,
#                                                save_weights_only = TRUE)
#     )
#   )

history <- segmentation_fit_generator(
  model = DCB2018_u_net,
  generator = train_DCB2018_generator,
  epochs = 100,
  steps_per_epoch = 21,
  model_filepath = here("development/data-science-bowl-2018/DSB2018_w.hdf5"))
```

Predict on new images:

``` r
DCB2018_u_net <- u_net(
  input_shape = c(net_h, net_w, 3),
  blocks = blocks,
  classes = classes,
  filters = 16,
  dropout = 0.1,
  batch_normalization = TRUE,
  kernel_initializer = "he_normal"
)
DCB2018_u_net %>% load_model_weights_hdf5(here("development/data-science-bowl-2018/DSB2018_w.hdf5"))

test_DCB2018_generator <- segmentation_generator(
  path = test_DCB2018_path, # directory with images and masks
  mode = "nested_dirs", # Each image with masks in separate folder
  classes = classes,
  only_images = TRUE,
  target_size = c(net_h, net_w),
  grayscale = FALSE,
  scale = 1 / 255,
  batch_size = 32,
  shuffle = FALSE,
  subdirs = c("images", "masks") # Names of subdirs with images and masks
)
```

    ## [1] "65 images detected!"

``` r
test_preds <- segmentation_predict_generator(DCB2018_u_net, test_DCB2018_generator, 3)
```
