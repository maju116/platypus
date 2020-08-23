
<img src="man/figures/hexsticker_platypus.png" align="right" alt="" width="130" />

platypus
========

**R package for object detection and image segmentation**

With `platypus` it is easy create advanced computer vision models like
YOLOv3 and U-Net in a few lines of code.

How to install?
---------------

You can install the latest version of `platypus` with `remotes`:

    remotes::install_github("maju116/platypus")

(`master` branch contains the stable version. Use `develop` branch for
latest features)

To install [previous versions]() you can run:

    remotes::install_github("maju116/platypus", ref = "0.1.0")

In order to install `platypus` you need to install `keras` and
`tensorflow` packages and `Tensorflow` version `>= 2.0.0`
(`Tensorflow 1.x` will not be supported!)

YOLOv3 Object detection with pre-trained COCO weights:
------------------------------------------------------

To create YOLOv3 architecture use:

    library(tidyverse)
    library(platypus)
    library(abind)

    test_yolo <- yolo3(
      net_h = 416, # Input image height
      net_w = 416, # Input image width
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

You can now load [YOLOv3 Darknet](https://pjreddie.com/darknet/yolo/)
weights trained on [COCO dataset](https://cocodataset.org/#home).
Download pre-trained weights from
[here](https://pjreddie.com/media/files/yolov3.weights) and run:

    test_yolo %>% load_darknet_weights("development/yolov3.weights")

Calculate predictions for new images:

    test_img_paths <- list.files(system.file("extdata", "images", package = "platypus"), full.names = TRUE)
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

Transform raw predictions into bounding boxes:

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
    #>        xmin      ymin      xmax      ymax     p_obj label_id  label
    #> 1 0.2065822 0.7178876 0.2364099 0.8651962 0.9507609        1 person
    #> 2 0.8122054 0.7581823 0.8457879 0.8682902 0.9587147        1 person
    #> 3 0.3485442 0.7018576 0.4919897 0.8842886 0.9997346        3    car
    #> 4 0.4835574 0.5431118 0.4976843 0.5575383 0.8372639        3    car
    #> 5 0.5016635 0.5427939 0.5154415 0.5558958 0.8209396        3    car
    #> 6 0.4394382 0.6041225 0.4685148 0.6429721 0.8417721        3    car
    #> 7 0.5407435 0.5537741 0.6669978 0.8086006 0.9994573        6    bus
    #> 8 0.5340163 0.5701706 0.6751449 0.8193321 0.9543661        7  train
    #> 
    #> [[2]]
    #>         xmin       ymin      xmax      ymax     p_obj label_id label
    #> 1 0.02362738 0.07048325 0.4544083 0.9091996 0.9999467       23 zebra
    #> 2 0.28961027 0.20582033 0.7285326 0.9007657 0.9972296       23 zebra
    #> 3 0.48607180 0.40705788 0.8476524 0.9278796 0.9995333       23 zebra

Plot / save images:

    plot_boxes(
      images_paths = test_img_paths, # Images paths
      boxes = test_boxes, # Bounding boxes
      correct_hw = TRUE # Should height and width of bounding boxes be corrected to image height and width
    )

![](man/figures/README-unnamed-chunk-8-1.png)<!-- -->![](man/figures/README-unnamed-chunk-8-2.png)<!-- -->
