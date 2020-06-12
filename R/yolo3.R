yolo3_conv2d <- function(inputs, filters) {
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
  keras_model(input, net_out)(inputs)
}

reshape_yolo3_output <- function(x, anchors, n_class) {
  x_shape <- x$get_shape()$as_list()
  k_reshape(x, list(-1, x_shape[[2]], x_shape[[3]],
                anchors, n_class + 5))
}

yolo3_output <- function(inputs, filters, anchors, n_class) {
  input <- layer_input(shape = inputs$get_shape()$as_list()[2:4])
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = filters * 2, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE) %>%
    darknet53_conv2d(strides = 1, filters = anchors * (n_class + 5), kernel_size = 1,
                     batch_normalization = FALSE, leaky_relu = FALSE)
  net_out <- layer_lambda(net_out, f = reshape_yolo3_output,
                          arguments = list(anchors = anchors, n_class = n_class))
  keras_model(input, net_out)(inputs)
}

yolo3 <- function(net_h = 416, net_w = 416, channels = 3, n_class = 80, anchors_per_grid = 3, anchors = coco_anchors) {
  input_img <- layer_input(shape = list(net_h, net_w, channels), name = 'input_img')
  darknet <- darknet53()(input_img)
  net_out <- yolo3_conv2d(darknet[[3]], 512)
  grid_13 <- yolo3_output(net_out, 512, anchors_per_grid, n_class)
  net_out <- yolo3_conv2d(list(net_out, darknet[[2]]), 256)
  grid_26 <- yolo3_output(net_out, 256, anchors_per_grid, n_class)
  net_out <- yolo3_conv2d(list(net_out, darknet[[1]]), 128)
  grid_52 <- yolo3_output(net_out, 128, anchors_per_grid, n_class)

  # grid_13_transform <- layer_lambda(grid_13, f = transform_boxes_tf,
  #                                   arguments = list(anchors = anchors[[1]], n_class = n_class,
  #                                                    net_h = net_h, net_w = net_w))
  # grid_26_transform <- layer_lambda(grid_26, f = transform_boxes_tf,
  #                                   arguments = list(anchors = anchors[[2]], n_class = n_class,
  #                                                    net_h = net_h, net_w = net_w))
  # grid_52_transform <- layer_lambda(grid_52, f = transform_boxes_tf,
  #                                   arguments = list(anchors = anchors[[3]], n_class = n_class,
  #                                                    net_h = net_h, net_w = net_w))

  keras_model(input_img, list(grid_13, grid_26, grid_52))
}

coco_anchors <- list(
  list(c(116, 90), c(156, 198), c(373, 326)),
  list(c(30, 61), c(62, 45), c(59, 119)),
  list(c(10, 13), c(16, 30), c(33, 23))
)

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
