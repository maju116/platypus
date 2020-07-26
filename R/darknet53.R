#' Creates a convolutional Darknet53 unit.
#' @description Creates a convolutional Darknet53 unit.
#' @import keras
#' @importFrom magrittr %>%
#' @importFrom purrr when
#' @param input Model or layer object.
#' @param strides An integer or list of 2 integers, specifying the strides of the convolution along the width and height.
#' @param filters Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#' @param kernel_size An integer or list of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
#' @param batch_normalization Should batch normalization be used in the unit.
#' @param leaky_relu Should leaky ReLU activation function be used in the unit.
#' @return Convolutional Darknet53 unit.
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

#' Creates a residual Darknet53 block.
#' @description Creates a residual Darknet53 block.
#' @import keras
#' @importFrom magrittr %>%
#' @param input Model or layer object.
#' @param filters Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#' @param blocks Number of residual blocks.
#' @return Residual Darknet53 block.
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

#' Creates a Darknet53 architecture.
#' @description Creates a Darknet53 architecture.
#' @import keras
#' @import tensorflow
#' @importFrom magrittr %>%
#' @param channels Number of channels.
#' @return Darknet53 model.
#' @export
darknet53 <- function(channels) {
  input <- layer_input(shape = list(NULL, NULL, channels))
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
  tf$keras$Model(input, list(out1, out2, net_out), name = "darknet53")
}
