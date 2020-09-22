Download images and annotations: [BCCD
dataset](https://www.kaggle.com/surajiiitm/bccd-dataset?).

Split dataset into train, validation and test set

``` r
library(tidyverse)
library(platypus)
library(abind)
library(here)

BCCD_path <- here("development/BCCD/")
annot_path <- file.path(BCCD_path, "Annotations/")
images_path <- file.path(BCCD_path, "JPEGImages/")

c("train", "valid", "test") %>%
  walk(~ {
    dir.create(file.path(BCCD_path, .))
    dir.create(file.path(BCCD_path, ., "Annotations/"))
    dir.create(file.path(BCCD_path, ., "JPEGImages/"))
  })

annot_paths <- list.files(annot_path, full.names = TRUE)
images_paths <- list.files(images_path, full.names = TRUE)
n_samples <- length(annot_paths)
set.seed(111)
train_ids <- sample(1:n_samples, round(0.8 * n_samples))
valid_ids <- sample(setdiff(1:n_samples, train_ids), round(0.19 * n_samples))
test_ids <- setdiff(1:n_samples, c(train_ids, valid_ids))

walk2(c("train", "valid", "test"), list(train_ids, valid_ids, test_ids), ~ {
  annots <- annot_paths[.y]
  images <- images_paths[.y]
  dir_name <- .x
  annots %>% walk(~ file.copy(., gsub("BCCD", paste0("BCCD/", dir_name), .)))
  images %>% walk(~ file.copy(., gsub("BCCD", paste0("BCCD/", dir_name), .)))
})
```

Generate custom anchor boxes:

``` r
blood_labels <- c("Platelets", "RBC", "WBC")
n_class <- length(blood_labels)
net_h <- 416 # Must be divisible by 32
net_w <- 416 # Must be divisible by 32
anchors_per_grid <- 3

blood_anchors <- generate_anchors(
  anchors_per_grid = anchors_per_grid, # Number of anchors (per one grid) to generate
  annot_path = annot_path, # Annotations directory
  labels = blood_labels, # Class labels
  n_iter = 10, # Number of k-means++ iterations
  annot_format = "pascal_voc", # Annotations format
  seed = 55, # Random seed
  centroid_fun = mean # Centroid function
)
```

    ##       label    n
    ## 1 Platelets  361
    ## 2       RBC 4153
    ## 3       WBC  372

![](Blood-Cell-Detection_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r
blood_anchors
```

    ## [[1]]
    ## [[1]][[1]]
    ## [1] 0.3552235 0.4417515
    ## 
    ## [[1]][[2]]
    ## [1] 0.2911290 0.3292675
    ## 
    ## [[1]][[3]]
    ## [1] 0.1971296 0.2346442
    ## 
    ## 
    ## [[2]]
    ## [[2]][[1]]
    ## [1] 0.1757463 0.1592062
    ## 
    ## [[2]][[2]]
    ## [1] 0.1652637 0.2065506
    ## 
    ## [[2]][[3]]
    ## [1] 0.1630269 0.2439239
    ## 
    ## 
    ## [[3]]
    ## [[3]][[1]]
    ## [1] 0.1391842 0.1769376
    ## 
    ## [[3]][[2]]
    ## [1] 0.1245985 0.2258089
    ## 
    ## [[3]][[3]]
    ## [1] 0.06237392 0.08062560

Build `YOLOv3` model (you can load [YOLOv3
Darknet](https://pjreddie.com/darknet/yolo/) weights trained on [COCO
dataset](https://cocodataset.org/#home). Download pre-trained weights
from [here](https://pjreddie.com/media/files/yolov3.weights)):

``` r
blood_yolo <- yolo3(
  net_h = net_h, # Input image height
  net_w = net_w, # Input image width
  grayscale = FALSE, # Should images be loaded as grayscale or RGB
  n_class = n_class, # Number of object classes (80 for COCO dataset)
  anchors = blood_anchors # Anchor boxes
)
blood_yolo %>% load_darknet_weights(here("development/yolov3.weights")) # Optional

blood_yolo
```

    ## Model
    ## Model: "yolo3"
    ## ________________________________________________________________________________
    ## Layer (type)              Output Shape      Param #  Connected to               
    ## ================================================================================
    ## input_img (InputLayer)    [(None, 416, 416, 0                                   
    ## ________________________________________________________________________________
    ## darknet53 (Model)         multiple          40620640 input_img[0][0]            
    ## ________________________________________________________________________________
    ## yolo3_conv1 (Model)       (None, 13, 13, 51 11024384 darknet53[1][2]            
    ## ________________________________________________________________________________
    ## yolo3_conv2 (Model)       (None, 26, 26, 25 2957312  yolo3_conv1[1][0]          
    ##                                                      darknet53[1][1]            
    ## ________________________________________________________________________________
    ## yolo3_conv3 (Model)       (None, 52, 52, 12 741376   yolo3_conv2[1][0]          
    ##                                                      darknet53[1][0]            
    ## ________________________________________________________________________________
    ## grid1 (Model)             (None, 13, 13, 3, 4747288  yolo3_conv1[1][0]          
    ## ________________________________________________________________________________
    ## grid2 (Model)             (None, 26, 26, 3, 1194008  yolo3_conv2[1][0]          
    ## ________________________________________________________________________________
    ## grid3 (Model)             (None, 52, 52, 3, 302104   yolo3_conv3[1][0]          
    ## ================================================================================
    ## Total params: 61,587,112
    ## Trainable params: 61,534,504
    ## Non-trainable params: 52,608
    ## ________________________________________________________________________________

Compile the model with correct loss and metrics:

``` r
blood_yolo %>% compile(
  optimizer = optimizer_adam(lr = 1e-5),
  loss = yolo3_loss(blood_anchors, n_class = n_class),
  metrics = yolo3_metrics(blood_anchors, n_class = n_class)
)
```

Create data generators:

``` r
train_blood_yolo_generator <- yolo3_generator(
  annot_path = file.path(BCCD_path, "train", "Annotations/"),
  images_path = file.path(BCCD_path, "train", "JPEGImages/"),
  net_h = net_h,
  net_w = net_w,
  batch_size = 16,
  shuffle = FALSE,
  labels = blood_labels
)
```

    ## 291 images with corresponding annotations detected!
    ## Set 'steps_per_epoch' to: 19

``` r
valid_blood_yolo_generator <- yolo3_generator(
  annot_path = file.path(BCCD_path, "valid", "Annotations/"),
  images_path = file.path(BCCD_path, "valid", "JPEGImages/"),
  net_h = net_h,
  net_w = net_w,
  batch_size = 16,
  shuffle = FALSE,
  labels = blood_labels
)
```

    ## 69 images with corresponding annotations detected!
    ## Set 'steps_per_epoch' to: 5

Fit the model:

``` r
blood_yolo %>%
  fit_generator(
    generator = blood_yolo_generator,
    epochs = 1000,
    steps_per_epoch = 19,
    validation_data = valid_blood_yolo_generator,
    validation_steps = 5,
    callbacks = list(callback_model_checkpoint("development/BCCD/blood_w.hdf5",
                                               save_best_only = TRUE,
                                               save_weights_only = TRUE)
    )
  )
```

Predict on new images:

``` r
blood_yolo <- yolo3(
  net_h = net_h,
  net_w = net_w,
  grayscale = FALSE,
  n_class = n_class,
  anchors = blood_anchors
)
blood_yolo %>% load_model_weights_hdf5(here("development/BCCD/blood_w.hdf5"))

test_blood_yolo_generator <- yolo3_generator(
  annot_path = file.path(BCCD_path, "test", "Annotations/"),
  images_path = file.path(BCCD_path, "test", "JPEGImages/"),
  net_h = net_h,
  net_w = net_w,
  batch_size = 4,
  shuffle = FALSE,
  labels = blood_labels
)
```

    ## 4 images with corresponding annotations detected!
    ## Set 'steps_per_epoch' to: 1

``` r
test_preds <- predict_generator(blood_yolo, test_blood_yolo_generator, 1)

test_boxes <- get_boxes(test_preds, blood_anchors, blood_labels,
                        obj_threshold = 0.6)

test_boxes
```

    ## [[1]]
    ## # A tibble: 17 x 7
    ##      xmin   ymin  xmax  ymax p_obj label_id label
    ##     <dbl>  <dbl> <dbl> <dbl> <dbl>    <int> <chr>
    ##  1 0.617  0.0137 0.884 0.543  1           2 RBC  
    ##  2 0.299  0      0.500 0.140  1.00        2 RBC  
    ##  3 0.479  0      0.687 0.203  1.00        2 RBC  
    ##  4 0.709  0      0.870 0.206  1.00        2 RBC  
    ##  5 0.0199 0      0.163 0.250  1.00        2 RBC  
    ##  6 0.0516 0.190  0.196 0.336  1           2 RBC  
    ##  7 0.406  0.164  0.564 0.316  1.00        2 RBC  
    ##  8 0.472  0.165  0.635 0.326  1.00        2 RBC  
    ##  9 0.865  0.0795 1     0.411  1.00        2 RBC  
    ## 10 0.0799 0.308  0.296 0.494  1.00        2 RBC  
    ## 11 0.414  0.359  0.562 0.506  1.00        2 RBC  
    ## 12 0.819  0.328  1     0.563  1.00        2 RBC  
    ## 13 0.482  0.395  0.623 0.544  1.00        2 RBC  
    ## 14 0.250  0.464  0.420 0.679  1           2 RBC  
    ## 15 0.868  0.500  1     0.704  1.00        2 RBC  
    ## 16 0.0584 0.531  0.290 0.821  1.00        3 WBC  
    ## 17 0.440  0.584  0.746 0.906  1.00        3 WBC  
    ## 
    ## [[2]]
    ## # A tibble: 16 x 7
    ##       xmin  ymin  xmax  ymax p_obj label_id label
    ##      <dbl> <dbl> <dbl> <dbl> <dbl>    <int> <chr>
    ##  1 0.133   0     0.409 0.287  1.00        2 RBC  
    ##  2 0.427   0.275 0.704 0.779  1.00        2 RBC  
    ##  3 0.831   0.104 0.970 0.285  1.00        2 RBC  
    ##  4 0.489   0.278 0.680 0.461  1.00        2 RBC  
    ##  5 0.00549 0.279 0.106 0.509  1.00        2 RBC  
    ##  6 0.579   0.306 0.757 0.496  1.00        2 RBC  
    ##  7 0.0502  0.341 0.204 0.572  1.00        2 RBC  
    ##  8 0.769   0.436 0.945 0.592  1.00        2 RBC  
    ##  9 0.360   0.536 0.510 0.683  1.00        2 RBC  
    ## 10 0.239   0.595 0.436 0.771  1.00        2 RBC  
    ## 11 0.626   0.678 0.812 0.878  1.00        2 RBC  
    ## 12 0.700   0.740 0.890 0.939  1.00        2 RBC  
    ## 13 0.805   0.715 0.985 0.915  1.00        2 RBC  
    ## 14 0.191   0.692 0.389 1      1.00        2 RBC  
    ## 15 0.347   0.756 0.544 1      1.00        2 RBC  
    ## 16 0.217   0.252 0.442 0.507  1.00        3 WBC  
    ## 
    ## [[3]]
    ## # A tibble: 13 x 7
    ##     xmin  ymin  xmax  ymax p_obj label_id label
    ##    <dbl> <dbl> <dbl> <dbl> <dbl>    <int> <chr>
    ##  1 0.593 0.106 0.838 0.617 0.642        2 RBC  
    ##  2 0.212 0.577 0.434 1     0.660        2 RBC  
    ##  3 0.280 0     0.453 0.236 1.00         2 RBC  
    ##  4 0.637 0     0.801 0.237 1.00         2 RBC  
    ##  5 0.780 0.184 0.972 0.399 0.996        2 RBC  
    ##  6 0.588 0.263 0.773 0.481 0.809        2 RBC  
    ##  7 0     0.342 0.214 0.514 0.931        2 RBC  
    ##  8 0.836 0.468 1     0.661 1.00         2 RBC  
    ##  9 0.778 0.686 0.969 0.873 1.00         2 RBC  
    ## 10 0.441 0.770 0.624 0.958 0.939        2 RBC  
    ## 11 0.610 0.783 0.806 0.969 1.00         2 RBC  
    ## 12 0.257 0.195 0.592 0.523 0.999        3 WBC  
    ## 13 0.225 0.259 0.512 0.559 0.999        3 WBC  
    ## 
    ## [[4]]
    ## # A tibble: 19 x 7
    ##       xmin   ymin   xmax  ymax p_obj label_id label    
    ##      <dbl>  <dbl>  <dbl> <dbl> <dbl>    <int> <chr>    
    ##  1 0.0478  0.239  0.219  0.336 0.982        1 Platelets
    ##  2 0.185   0.512  0.305  0.685 0.978        1 Platelets
    ##  3 0.318   0.578  0.496  0.689 0.998        1 Platelets
    ##  4 0       0.510  0.0972 0.777 1.00         1 Platelets
    ##  5 0.158   0      0.343  0.281 0.947        2 RBC      
    ##  6 0.874   0.0645 1      0.235 0.962        2 RBC      
    ##  7 0.372   0.151  0.567  0.325 1.00         2 RBC      
    ##  8 0.549   0.155  0.709  0.478 1.00         2 RBC      
    ##  9 0.445   0.251  0.607  0.566 0.984        2 RBC      
    ## 10 0.458   0.339  0.639  0.514 0.988        2 RBC      
    ## 11 0.00327 0.466  0.193  0.642 1.00         2 RBC      
    ## 12 0.463   0.562  0.642  0.789 1.00         2 RBC      
    ## 13 0.178   0.671  0.331  0.847 0.997        2 RBC      
    ## 14 0.148   0.671  0.370  0.885 0.975        2 RBC      
    ## 15 0.358   0.696  0.546  1     0.994        2 RBC      
    ## 16 0.535   0.806  0.710  0.969 1.00         2 RBC      
    ## 17 0.692   0.705  0.892  1     0.999        2 RBC      
    ## 18 0.837   0.783  1      1     0.994        2 RBC      
    ## 19 0.615   0.408  0.904  0.748 1.00         3 WBC

Plot / save images with predicted bounding boxes:

``` r
plot_boxes(
  images_paths = list.files(file.path(BCCD_path, "test", "JPEGImages/"), full.names = TRUE),
  boxes = test_boxes,
  labels = blood_labels,
  save_dir = BCCD_path)
```

![](Blood-Cell-Detection_files/figure-markdown_github/unnamed-chunk-8-1.png)![](Blood-Cell-Detection_files/figure-markdown_github/unnamed-chunk-8-2.png)![](Blood-Cell-Detection_files/figure-markdown_github/unnamed-chunk-8-3.png)![](Blood-Cell-Detection_files/figure-markdown_github/unnamed-chunk-8-4.png)
