<img src="man/figures/hexsticker_platypus.png" align="right" alt="" width="130" />

platypus
========

**R package for object detection and image segmentation**

With `platypus` it is easy create advanced computer vision models like
YOLOv3 and U-Net in a few lines of code.

How to install?
---------------

You can install the latest version of `platypus` with `remotes`:

``` r
remotes::install_github("maju116/platypus")
```

(`master` branch contains the stable version. Use `develop` branch for
latest features)

To install [previous versions]() you can run:

``` r
remotes::install_github("maju116/platypus", ref = "0.1.0")
```

In order to install `platypus` you need to install `keras` and
`tensorflow` packages and `Tensorflow` version `>= 2.0.0`
(`Tensorflow 1.x` will not be supported!)

YOLOv3 bounding box prediction with pre-trained COCO weights:
-------------------------------------------------------------

To create `YOLOv3` architecture use:

``` r
library(tidyverse)
library(platypus)
library(abind)

test_yolo <- yolo3(
  net_h = 416, # Input image height. Must be divisible by 32
  net_w = 416, # Input image width. Must be divisible by 32
  grayscale = FALSE, # Should images be loaded as grayscale or RGB
  n_class = 80, # Number of object classes (80 for COCO dataset)
  anchors = coco_anchors # Anchor boxes
)

test_yolo
#> Model
#> Model: "yolo3"
#> ________________________________________________________________________________
#> Layer (type)              Output Shape      Param #  Connected to               
#> ================================================================================
#> input_img (InputLayer)    [(None, 416, 416, 0                                   
#> ________________________________________________________________________________
#> darknet53 (Model)         multiple          40620640 input_img[0][0]            
#> ________________________________________________________________________________
#> yolo3_conv1 (Model)       (None, 13, 13, 51 11024384 darknet53[1][2]            
#> ________________________________________________________________________________
#> yolo3_conv2 (Model)       (None, 26, 26, 25 2957312  yolo3_conv1[1][0]          
#>                                                      darknet53[1][1]            
#> ________________________________________________________________________________
#> yolo3_conv3 (Model)       (None, 52, 52, 12 741376   yolo3_conv2[1][0]          
#>                                                      darknet53[1][0]            
#> ________________________________________________________________________________
#> grid1 (Model)             (None, 13, 13, 3, 4984063  yolo3_conv1[1][0]          
#> ________________________________________________________________________________
#> grid2 (Model)             (None, 26, 26, 3, 1312511  yolo3_conv2[1][0]          
#> ________________________________________________________________________________
#> grid3 (Model)             (None, 52, 52, 3, 361471   yolo3_conv3[1][0]          
#> ================================================================================
#> Total params: 62,001,757
#> Trainable params: 61,949,149
#> Non-trainable params: 52,608
#> ________________________________________________________________________________
```

You can now load [YOLOv3 Darknet](https://pjreddie.com/darknet/yolo/)
weights trained on [COCO dataset](https://cocodataset.org/#home).
Download pre-trained weights from
[here](https://pjreddie.com/media/files/yolov3.weights) and run:

``` r
test_yolo %>% load_darknet_weights("development/yolov3.weights")
```

Calculate predictions for new images:

``` r
test_img_paths <- list.files(system.file("extdata", "images", package = "platypus"), full.names = TRUE, pattern = "coco")
test_imgs <- test_img_paths %>%
  map(~ {
    image_load(., target_size = c(416, 416), grayscale = FALSE) %>%
      image_to_array() %>%
      `/`(255)
  }) %>%
  abind(along = 4) %>%
  aperm(c(4, 1:3))
test_preds <- test_yolo %>% predict(test_imgs)

str(test_preds)
#> List of 3
#>  $ : num [1:2, 1:13, 1:13, 1:3, 1:85] 0.294 0.478 0.371 1.459 0.421 ...
#>  $ : num [1:2, 1:26, 1:26, 1:3, 1:85] -0.214 1.093 -0.092 2.034 -0.286 ...
#>  $ : num [1:2, 1:52, 1:52, 1:3, 1:85] 0.242 -0.751 0.638 -2.419 -0.282 ...
```

Transform raw predictions into bounding boxes:

``` r
test_boxes <- get_boxes(
  preds = test_preds, # Raw predictions form YOLOv3 model
  anchors = coco_anchors, # Anchor boxes
  labels = coco_labels, # Class labels
  obj_threshold = 0.6, # Object threshold
  nms = TRUE, # Should non-max suppression be applied
  nms_threshold = 0.6, # Non-max suppression threshold
  correct_hw = FALSE # Should height and width of bounding boxes be corrected to image height and width
)

test_boxes
#> [[1]]
#> # A tibble: 8 x 7
#>    xmin  ymin  xmax  ymax p_obj label_id label 
#>   <dbl> <dbl> <dbl> <dbl> <dbl>    <int> <chr> 
#> 1 0.207 0.718 0.236 0.865 0.951        1 person
#> 2 0.812 0.758 0.846 0.868 0.959        1 person
#> 3 0.349 0.702 0.492 0.884 1.00         3 car   
#> 4 0.484 0.543 0.498 0.558 0.837        3 car   
#> 5 0.502 0.543 0.515 0.556 0.821        3 car   
#> 6 0.439 0.604 0.469 0.643 0.842        3 car   
#> 7 0.541 0.554 0.667 0.809 0.999        6 bus   
#> 8 0.534 0.570 0.675 0.819 0.954        7 train 
#> 
#> [[2]]
#> # A tibble: 3 x 7
#>     xmin   ymin  xmax  ymax p_obj label_id label
#>    <dbl>  <dbl> <dbl> <dbl> <dbl>    <int> <chr>
#> 1 0.0236 0.0705 0.454 0.909 1.00        23 zebra
#> 2 0.290  0.206  0.729 0.901 0.997       23 zebra
#> 3 0.486  0.407  0.848 0.928 1.00        23 zebra
```

Plot / save images:

``` r
plot_boxes(
  images_paths = test_img_paths, # Images paths
  boxes = test_boxes, # Bounding boxes
  correct_hw = TRUE, # Should height and width of bounding boxes be corrected to image height and width
  labels = coco_labels # Class labels
)
```

![](man/figures/README-unnamed-chunk-8-1.png)![](man/figures/README-unnamed-chunk-8-2.png)

YOLOv3 Object detection with custom dataset:
--------------------------------------------

Download images and annotations: [BCCD
dataset](https://www.kaggle.com/surajiiitm/bccd-dataset?)

Generate custom anchor boxes:

``` r
library(tidyverse)
library(platypus)
library(abind)

BCCD_path <- "development/BCCD/"
annot_path <- file.path(BCCD_path, "Annotations/")
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
#>       label    n
#> 1 Platelets  361
#> 2       RBC 4153
#> 3       WBC  372
```

![](man/figures/README-unnamed-chunk-9-1.png)

``` r
blood_anchors
#> [[1]]
#> [[1]][[1]]
#> [1] 0.3552235 0.4417515
#> 
#> [[1]][[2]]
#> [1] 0.2911290 0.3292675
#> 
#> [[1]][[3]]
#> [1] 0.1971296 0.2346442
#> 
#> 
#> [[2]]
#> [[2]][[1]]
#> [1] 0.1757463 0.1592062
#> 
#> [[2]][[2]]
#> [1] 0.1652637 0.2065506
#> 
#> [[2]][[3]]
#> [1] 0.1630269 0.2439239
#> 
#> 
#> [[3]]
#> [[3]][[1]]
#> [1] 0.1391842 0.1769376
#> 
#> [[3]][[2]]
#> [1] 0.1245985 0.2258089
#> 
#> [[3]][[3]]
#> [1] 0.06237392 0.08062560
```

Build `YOLOv3` model and compile it with correct loss and metric:

``` r
blood_yolo <- yolo3(
  net_h = net_h, # Input image height
  net_w = net_w, # Input image width
  grayscale = FALSE, # Should images be loaded as grayscale or RGB
  n_class = n_class, # Number of object classes (80 for COCO dataset)
  anchors = blood_anchors # Anchor boxes
)
blood_yolo %>% load_darknet_weights("development/yolov3.weights") # Optional

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
#> 291 images with corresponding annotations detected!
#> Set 'steps_per_epoch' to: 19

valid_blood_yolo_generator <- yolo3_generator(
  annot_path = file.path(BCCD_path, "valid", "Annotations/"),
  images_path = file.path(BCCD_path, "valid", "JPEGImages/"),
  net_h = net_h,
  net_w = net_w,
  batch_size = 16,
  shuffle = FALSE,
  labels = blood_labels
)
#> 69 images with corresponding annotations detected!
#> Set 'steps_per_epoch' to: 5
```

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
blood_yolo %>% load_model_weights_hdf5("development/BCCD/blood_w.hdf5")

test_blood_yolo_generator <- yolo3_generator(
  annot_path = file.path(BCCD_path, "test", "Annotations/"),
  images_path = file.path(BCCD_path, "test", "JPEGImages/"),
  net_h = net_h,
  net_w = net_w,
  batch_size = 4,
  shuffle = FALSE,
  labels = blood_labels
)
#> 4 images with corresponding annotations detected!
#> Set 'steps_per_epoch' to: 1

test_preds <- predict_generator(blood_yolo, test_blood_yolo_generator, 1)

test_boxes <- get_boxes(test_preds, blood_anchors, blood_labels,
                        obj_threshold = 0.6)

plot_boxes(
  images_paths = list.files(file.path(BCCD_path, "test", "JPEGImages/"), full.names = TRUE),
  boxes = test_boxes,
  labels = blood_labels)
```

![](man/figures/README-unnamed-chunk-13-1.png)![](man/figures/README-unnamed-chunk-13-2.png)![](man/figures/README-unnamed-chunk-13-3.png)![](man/figures/README-unnamed-chunk-13-4.png)

See full example
[here](https://github.com/maju116/platypus/blob/master/examples/Blood%20Cell%20Detection/Blood-Cell-Detection.md)

U-Net image segmentation with custom dataset:
---------------------------------------------

Build `U-Net` model and compile it with correct loss and metric:

``` r
library(tidyverse)
library(platypus)
library(abind)

train_DCB2018_path <- "development/data-science-bowl-2018/stage1_train"
test_DCB2018_path <- "development/data-science-bowl-2018/stage1_test"

blocks <- 4 # Number of U-Net convolutional blocks
n_class <- 2 # Number of classes
net_h <- 256 # Must be in a form of 2^N
net_w <- 256 # Must be in a form of 2^N

DCB2018_u_net <- u_net(
  net_h = net_h,
  net_w = net_w,
  grayscale = FALSE,
  blocks = blocks,
  n_class = n_class,
  filters = 16,
  dropout = 0.1,
  batch_normalization = TRUE,
  kernel_initializer = "he_normal"
)

DCB2018_u_net %>%
  compile(
    optimizer = optimizer_adam(lr = 1e-3),
    loss = loss_dice(),
    metrics = metric_dice_coeff()
  )
```

Create data generator:

``` r
train_DCB2018_generator <- segmentation_generator(
  path = train_DCB2018_path, # directory with images and masks
  mode = "nested_dirs", # Each image with masks in separate folder
  colormap = binary_colormap,
  only_images = FALSE,
  net_h = net_h,
  net_w = net_w,
  grayscale = FALSE,
  scale = 1 / 255,
  batch_size = 32,
  shuffle = TRUE,
  subdirs = c("images", "masks") # Names of subdirs with images and masks
)
#> 670 images with corresponding masks detected!
#> Set 'steps_per_epoch' to: 21
```

Fit the model:

``` r
history <- DCB2018_u_net %>%
  fit_generator(
    train_DCB2018_generator,
    epochs = 20,
    steps_per_epoch = 21,
    callbacks = list(callback_model_checkpoint(
      "development/data-science-bowl-2018/DSB2018_w.hdf5",
      save_best_only = TRUE,
      save_weights_only = TRUE,
      monitor = "dice_coeff",
      mode = "max",
      verbose = 1)
    )
  )
```

Predict on new images:

``` r
DCB2018_u_net <- u_net(
  net_h = net_h,
  net_w = net_w,
  grayscale = FALSE,
  blocks = blocks,
  filters = 16,
  dropout = 0.1,
  batch_normalization = TRUE,
  kernel_initializer = "he_normal"
)
DCB2018_u_net %>% load_model_weights_hdf5("development/data-science-bowl-2018/DSB2018_w.hdf5")

test_DCB2018_generator <- segmentation_generator(
  path = test_DCB2018_path,
  mode = "nested_dirs",
  colormap = binary_colormap,
  only_images = TRUE,
  net_h = net_h,
  net_w = net_w,
  grayscale = FALSE,
  scale = 1 / 255,
  batch_size = 32,
  shuffle = FALSE,
  subdirs = c("images", "masks")
)
#> 65 images detected!
#> Set 'steps_per_epoch' to: 3

test_preds <- predict_generator(DCB2018_u_net, test_DCB2018_generator, 3)

test_masks <- get_masks(test_preds, binary_colormap)
```

Plot / save images with masks:

``` r
test_imgs_paths <- create_images_masks_paths(test_DCB2018_path, "nested_dirs", FALSE, c("images", "masks"), ";")$images_paths

plot_masks(
  images_paths = test_imgs_paths[1:4],
  masks = test_masks[1:4],
  labels = c("background", "nuclei"),
  colormap = binary_colormap
)
```

![](man/figures/README-unnamed-chunk-18-1.png)![](man/figures/README-unnamed-chunk-18-2.png)![](man/figures/README-unnamed-chunk-18-3.png)![](man/figures/README-unnamed-chunk-18-4.png)

See full example
[here](https://github.com/maju116/platypus/blob/master/examples/2018%20Data%20Science%20Bowl/2018-Data-Science-Bowl.md)
