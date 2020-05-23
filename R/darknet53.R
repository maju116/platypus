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
