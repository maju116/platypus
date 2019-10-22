
yolo3_conv2d <- function(input, skip_connection = FALSE, strides, filters, kernel_size,
                         kernel_initializer = "he_normal", id,
                         batch_normalization = TRUE, leaky_relu = TRUE) {
  input %>%
    when(strides > 1 ~ layer_zero_padding_2d(., list(c(1, 0), c(1, 0))), ~ .) %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size, strides = strides,
                  padding = if (strides > 1) "valid" else "same",
                  name = paste0("conv2d_", id),
                  use_bias = if (batch_normalization) FALSE else TRUE,
                  kernel_initializer = kernel_initializer) %>%
    when(batch_normalization ~ layer_batch_normalization(., name = paste0("batch_norm_", id)), ~ .) %>%
    when(leaky_relu ~ layer_activation_leaky_relu(., alpha = 0.1, name = paste0("leaky_relu_", id)), ~ .) %>%
    when(skip_connection ~ layer_add(list(input, .), name = paste0("leaky_relu_", id)), ~ .)
}

yolo3 <- function(input_shape) {
  input_img <- layer_input(shape = input_shape, name = 'input_img') %>%
    yolo3_conv2d(skip_connection = FALSE, strides = 1, filters = 32,
                 kernel_size = 3, id = 0,  leaky_relu = TRUE)
}
