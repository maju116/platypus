#' Creates a convolutional Yolo3 unit.
#' @description Creates a convolutional Yolo3 unit.
#' @import keras
#' @import tensorflow
#' @importFrom magrittr %>%
#' @param inputs Models or layer objects.
#' @param filters Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#' @param name Model name.
#' @return Convolutional Yolo3 unit.
yolo3_conv2d <- function(inputs, filters, name) {
  if (is.list(inputs)) {
    input1 <- layer_input(shape = inputs[[1]]$get_shape()$as_list()[2:4])
    input2 <- layer_input(shape = inputs[[2]]$get_shape()$as_list()[2:4])
    input <- list(input1, input2)
    net_out <- input1 %>%
      darknet53_conv2d(strides = 1, filters = filters, kernel_size = 1,
                       batch_normalization = TRUE, leaky_relu = TRUE) %>%
      layer_upsampling_2d(size = 2)
    net_out <- layer_concatenate(list(net_out, input2))
  } else {
    input <- layer_input(shape = inputs$get_shape()$as_list()[2:4])
    net_out <- input
  }
  net_out <- net_out %>%
    darknet53_conv2d(strides = 1, filters = filters, kernel_size = 1,
                     batch_normalization = TRUE, leaky_relu = TRUE) %>%
    darknet53_conv2d(strides = 1, filters = filters * 2, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE) %>%
    darknet53_conv2d(strides = 1, filters = filters, kernel_size = 1,
                     batch_normalization = TRUE, leaky_relu = TRUE) %>%
    darknet53_conv2d(strides = 1, filters = filters * 2, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE) %>%
    darknet53_conv2d(strides = 1, filters = filters, kernel_size = 1,
                     batch_normalization = TRUE, leaky_relu = TRUE)
  tf$keras$Model(input, net_out, name = name)(inputs)
}

#' Creates Yolo3 output grid.
#' @description Creates Yolo3 output grid of dimensionality `(S, H, W, anchors_per_grid, 5 + n_class)`.
#' @import keras
#' @import tensorflow
#' @param inputs Models or layer objects.
#' @param filters Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#' @param anchors_per_grid Number of anchors/boxes per one output grid.
#' @param n_class Number of prediction classes.
#' @param name Model name.
#' @return Yolo3 output grid.
yolo3_output <- function(inputs, filters, anchors_per_grid, n_class, name) {
  input <- layer_input(shape = inputs$get_shape()$as_list()[2:4])
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = filters * 2, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE) %>%
    darknet53_conv2d(strides = 1, filters = anchors_per_grid * (n_class + 5), kernel_size = 1,
                     batch_normalization = FALSE, leaky_relu = FALSE)
  net_out_shape <- net_out$get_shape()$as_list()
  net_out <- layer_reshape(
    net_out,
    target_shape = c(net_out_shape[[2]], net_out_shape[[3]],
                     anchors_per_grid, n_class + 5))
  tf$keras$Model(input, net_out, name = name)(inputs)
}

#' Creates a Yolo3 architecture.
#' @description Creates a Yolo3 architecture.
#' @import keras
#' @import tensorflow
#' @param net_h Input layer height. Must be divisible by `32`.
#' @param net_w Input layer width. Must be divisible by `32`.
#' @param grayscale Defines input layer color channels -  `1` if `TRUE`, `3` if `FALSE`.
#' @param n_class Number of prediction classes.
#' @param anchors Prediction anchors. For exact format check \code{\link[platypus]{coco_anchors}}.
#' @return Yolo3 model.
#' @export
yolo3 <- function(net_h = 416, net_w = 416, grayscale = FALSE, n_class = 80, anchors = coco_anchors) {
  anchors_per_grid <- length(anchors[[1]])
  channels <- if (grayscale) 1 else 3
  input_img <- layer_input(shape = list(net_h, net_w, channels), name = 'input_img')
  darknet <- darknet53()(input_img)
  net_out <- yolo3_conv2d(darknet[[3]], 512, name = "yolo3_conv1")
  grid_1 <- yolo3_output(net_out, 512, anchors_per_grid, n_class, name = "grid1")
  net_out <- yolo3_conv2d(list(net_out, darknet[[2]]), 256, name = "yolo3_conv2")
  grid_2 <- yolo3_output(net_out, 256, anchors_per_grid, n_class, name = "grid2")
  net_out <- yolo3_conv2d(list(net_out, darknet[[1]]), 128, name = "yolo3_conv3")
  grid_3 <- yolo3_output(net_out, 128, anchors_per_grid, n_class, name = "grid3")
  tf$keras$Model(input_img, list(grid_1, grid_2, grid_3), name = "yolo3")
}

#' COCO dataset anchors.
#' @description COCO dataset anchors.
#' @return COCO dataset anchors.
#' @export
coco_anchors <- list(
  list(c(116, 90) / 416, c(156, 198) / 416, c(373, 326) / 416),
  list(c(30, 61) / 416, c(62, 45) / 416, c(59, 119) / 416),
  list(c(10, 13) / 416, c(16, 30) / 416, c(33, 23) / 416)
)

#' COCO dataset labels.
#' @description COCO dataset labels.
#' @return COCO dataset labels.
#' @export
coco_labels = c("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
