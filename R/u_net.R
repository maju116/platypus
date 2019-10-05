#' Creates a downsampling U-Net block.
#' @description Creates a downsampling U-Net block.
#' @import keras
#' @importFrom magrittr %>%
#' @importFrom purrr when
#' @param input Model or layer object
#' @param filters Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#' @param kernel_size An integer or list of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
#' @param batch_normalization Shoud batch normalization be used in the block.
#' @return Downsalmling U-Net block
#' @export
u_net_down <- function(input, filters, kernel_size, batch_normalization = TRUE) {
  # First layer
  input %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size,
                  padding = "same", kernel_initializer = "he_normal") %>%
    when(batch_normalization ~ layer_batch_normalization(.), ~ .) %>%
    layer_activation_relu() %>%
    # Second layer
    layer_conv_2d(filters = filters, kernel_size = kernel_size,
                  padding = "same", kernel_initializer = "he_normal") %>%
    when(batch_normalization ~ layer_batch_normalization(.), ~ .) %>%
    layer_activation_relu()
}

#' Creates a U-Net architecture.
#' @description Creates a U-Net architecture.
#' @import keras
#' @importFrom magrittr %>%
#' @importFrom purrr when
#' @param input_img Model or layer object
#' @param filters Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#' @param dropout Dropout rate.
#' @param batch_normalization Shoud batch normalization be used in the block.
#' @return Downsalmling U-Net block
#' @export
u_net <- function(input_img, filters, dropout = 0.1, batch_normalization = TRUE) {
  c1 <- u_net_down(input_img, filters * 1, kernel_size = 3, batchnorm = batchnorm)
  p1 <- layer_max_pooling_2d(c1, pool_size = 2) %>%
    layer_dropout(rate = dropout)

  c2 <- u_net_down(p1, filters * 2, kernel_size = 3, batchnorm = batchnorm)
  p2 <- layer_max_pooling_2d(c2, pool_size = 2) %>%
    layer_dropout(rate = dropout)

  c3 <- u_net_down(p2, filters * 4, kernel_size = 3, batchnorm = batchnorm)
  p3 <- layer_max_pooling_2d(c3, pool_size = 2) %>%
    layer_dropout(rate = dropout)

  c4 <- u_net_down(p3, filters * 8, kernel_size = 3, batchnorm = batchnorm)
  p4 <- layer_max_pooling_2d(c4, pool_size = 2) %>%
    layer_dropout(rate = dropout)

  c5 <- u_net_down(p4, filters * 16, kernel_size = 3, batchnorm = batchnorm)

  u6 <- layer_conv_2d_transpose(c5, filters * 8, kernel_size = 3, strides = 2) %>%
    layer_concatenate(inputs = list(., c4)) %>%
    layer_dropout(rate = dropout)
  c6 <- u_net_down(u6, filters * 8, kernel_size = 3, batchnorm = batchnorm)

  u7 <- layer_conv_2d_transpose(c6, filters * 4, kernel_size = 3, strides = 2) %>%
    layer_concatenate(inputs = list(., c3)) %>%
    layer_dropout(rate = dropout)
  c7 <- u_net_down(u7, filters * 4, kernel_size = 3, batchnorm = batchnorm)

  u8 <- layer_conv_2d_transpose(c7, filters * 2, kernel_size = 3, strides = 2) %>%
    layer_concatenate(inputs = list(., c2)) %>%
    layer_dropout(rate = dropout)
  c8 <- u_net_down(u8, filters * 2, kernel_size = 3, batchnorm = batchnorm)

  u9 <- layer_conv_2d_transpose(c8, filters * 1, kernel_size = 3, strides = 2) %>%
    layer_concatenate(inputs = list(., c1)) %>%
    layer_dropout(rate = dropout)
  c9 <- u_net_down(u9, filters * 1, kernel_size = 3, batchnorm = batchnorm)

  output <- layer_conv_2d(c9, 1, 1, activation = "sigmoid")
  keras_model(inputs = input_img, outputs = output)
}

