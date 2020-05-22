darknet53_conv2d <- function(input, strides, filters, kernel_size,
                             batch_normalization = TRUE, leaky_relu = TRUE) {
  input %>%
    when(strides > 1 ~ layer_zero_padding_2d(., list(c(1, 0), c(1, 0))), ~ .) %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size, strides = strides,
                  padding = if (strides > 1) "valid" else "same",
                  use_bias = if (batch_normalization) FALSE else TRUE,
                  kernel_initializer = initializer_glorot_uniform(),
                  kernel_regularizer = regularizer_l2(l = 5e-4)) %>%
    when(batch_normalization ~ layer_batch_normalization(., center = TRUE, scale = TRUE,
                                                         momentum = 0.99, epsilon = 1e-3), ~ .) %>%
    when(leaky_relu ~ layer_activation_leaky_relu(., alpha = 0.1), ~ .)
}

darknet53_residual_block <- function(input, filters, blocks) {
  net_out <- input %>%
    darknet53_conv2d(strides = 2, filters = filters, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE)
  for (block in 1:blocks) {
    add_layer <- net_out
    net_out <- net_out %>%
      darknet53_conv2d(strides = 1, filters = round(filters / 2), kernel_size = 1,
                       batch_normalization = TRUE, leaky_relu = TRUE) %>%
      darknet53_conv2d(strides = 1, filters = filters, kernel_size = 3,
                       batch_normalization = TRUE, leaky_relu = TRUE)
    net_out <- layer_add(list(add_layer, net_out))
  }
  net_out
}

darknet53 <- function() {
  input <- layer_input(shape = list(NULL, NULL, 3))
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = 32, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE) %>%
    darknet53_residual_block(filters = 64, blocks = 1) %>%
    darknet53_residual_block(filters = 128, blocks = 2) %>%
    darknet53_residual_block(filters = 256, blocks = 8)
  out1 <- net_out
  net_out <- net_out %>% darknet53_residual_block(filters = 512, blocks = 8)
  out2 <- net_out
  net_out <- net_out %>% darknet53_residual_block(filters = 1024, blocks = 4)
  keras_model(input, list(out1, out2, net_out))
}

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

yolo3_output <- function(inputs, filters, anchors, classes) {
  input <- layer_input(shape = inputs$get_shape()$as_list()[2:4])
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = filters * 2, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE) %>%
    darknet53_conv2d(strides = 1, filters = anchors * (classes + 5), kernel_size = 1,
                     batch_normalization = FALSE, leaky_relu = FALSE)
  # Add lambda fun
  keras_model(input, net_out)(inputs)
}

yolo3 <- function(input_shape = 416, channels = 3, classes = 80, anchors_per_grid = 3) {
  input_img <- layer_input(shape = c(input_shape, input_shape, channels), name = 'input_img')
  darknet <- darknet53()(input_img)
  net_out <- yolo3_conv2d(darknet[[3]], 512)
  grid_13 <- yolo3_output(net_out, 512, anchors_per_grid, classes)
  net_out <- yolo3_conv2d(list(net_out, darknet[[2]]), 256)
  grid_26 <- yolo3_output(net_out, 256, anchors_per_grid, classes)
  net_out <- yolo3_conv2d(list(net_out, darknet[[1]]), 128)
  grid_52 <- yolo3_output(net_out, 128, anchors_per_grid, classes)
  keras_model(input_img, list(grid_13, grid_26, grid_52))
}
