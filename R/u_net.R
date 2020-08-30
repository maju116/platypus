#' Creates a double convolutional U-Net block.
#' @description Creates a double convolutional U-Net block.
#' @import keras
#' @importFrom magrittr %>%
#' @importFrom purrr when
#' @param input Model or layer object.
#' @param filters Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#' @param kernel_size An integer or list of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
#' @param batch_normalization Should batch normalization be used in the block.
#' @param kernel_initializer Initializer for the kernel weights matrix.
#' @return Double convolutional U-Net block.
u_net_double_conv2d <- function(input, filters, kernel_size, batch_normalization = TRUE, kernel_initializer = "he_normal") {
  input %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size,
                  padding = "same", kernel_initializer = kernel_initializer) %>%
    when(batch_normalization ~ layer_batch_normalization(.), ~ .) %>%
    layer_activation_relu() %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size,
                  padding = "same", kernel_initializer = kernel_initializer) %>%
    when(batch_normalization ~ layer_batch_normalization(.), ~ .) %>%
    layer_activation_relu()
}

#' Creates a U-Net architecture.
#' @description Creates a U-Net architecture.
#' @import keras
#' @importFrom magrittr %>%
#' @importFrom purrr when
#' @param net_h Input layer height. Must be equal to `2^x, x - natural`..
#' @param net_w Input layer width. Must be equal to `2^x, x - natural`.
#' @param grayscale Defines input layer color channels -  `1` if `TRUE`, `3` if `FALSE`.
#' @param blocks Number of blocks in the model.
#' @param n_class Number of classes. Minimum is `2` (background + other object).
#' @param filters Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#' @param dropout Dropout rate.
#' @param batch_normalization Should batch normalization be used in the block.
#' @param kernel_initializer Initializer for the kernel weights matrix.
#' @return U-Net model.
#' @export
u_net <- function(net_h, net_w, grayscale, blocks = 4, n_class = 2, filters = 16,
                  dropout = 0.1, batch_normalization = TRUE, kernel_initializer = "he_normal") {
  channels <- if (grayscale) 1 else 3
  input_shape <- c(net_h, net_w, channels)
  u_net_check(input_shape, blocks, n_class, filters, dropout, batch_normalization)
  input_img <- layer_input(shape = input_shape, name = 'input_img')

  conv_layers <- pool_layers <- conv_tr_layers <- list()

  for (block in 1:blocks) {
    current_input <- if (block == 1) input_img else pool_layers[[block - 1]]
    conv_layers[[block]] <- u_net_double_conv2d(current_input, filters * 2^(block - 1), kernel_size = 3,
                                                batch_normalization = batch_normalization,
                                                kernel_initializer = kernel_initializer)
    pool_layers[[block]] <- layer_max_pooling_2d(conv_layers[[block]], pool_size = 2) %>%
      layer_dropout(rate = dropout)
  }

  conv_layers[[blocks + 1]] <- u_net_double_conv2d(pool_layers[[blocks]], filters * 2^blocks, kernel_size = 3,
                                                   batch_normalization = batch_normalization,
                                                   kernel_initializer = kernel_initializer)

  for (block in 1:blocks) {
    conv_tr_layers[[block]] <- layer_conv_2d_transpose(conv_layers[[blocks + block]], filters * 2^(blocks - block), kernel_size = 3,
                                                       strides = 2, padding = "same")
    conv_tr_layers[[block]] <- layer_concatenate(inputs = list(conv_tr_layers[[block]], conv_layers[[blocks - block + 1]])) %>%
      layer_dropout(rate = dropout)
    conv_layers[[blocks + block + 1]] <- u_net_double_conv2d(conv_tr_layers[[block]], filters * 2^(blocks - block), kernel_size = 3,
                                                             batch_normalization = batch_normalization,
                                                             kernel_initializer = kernel_initializer)
  }

  output <- layer_conv_2d(conv_layers[[2 * blocks + 1]], n_class, 1, activation = "softmax")
  keras_model(inputs = input_img, outputs = output)
}
