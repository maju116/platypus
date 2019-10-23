yolo3_conv2d <- function(input, skip_connection = NULL, strides, filters, kernel_size,
                         kernel_initializer = "he_normal", id,
                         batch_normalization = TRUE, leaky_relu = TRUE,
                         concatenate = NULL) {
  input %>%
    when(strides > 1 ~ layer_zero_padding_2d(., list(c(1, 0), c(1, 0))), ~ .) %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size, strides = strides,
                  padding = if (strides > 1) "valid" else "same",
                  name = paste0("conv2d_", id),
                  use_bias = if (batch_normalization) FALSE else TRUE,
                  kernel_initializer = kernel_initializer) %>%
    when(batch_normalization ~ layer_batch_normalization(., name = paste0("batch_norm_", id)), ~ .) %>%
    when(leaky_relu ~ layer_activation_leaky_relu(., alpha = 0.1, name = paste0("leaky_relu_", id)), ~ .) %>%
    when(!is.null(skip_connection) ~ layer_add(list(skip_connection, .), name = paste0("add_", id)), ~ .) %>%
    when(!is.null(concatenate) ~ layer_upsampling_2d(., size = 2, name = paste0("upsample_", id)), ~ .) %>%
    when(!is.null(concatenate) ~ layer_concatenate(list(concatenate, .), name = paste0("concat_", id)), ~ .)
}

yolo3_config <- list(
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 32, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 2, filters = 64, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 32, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 2, strides = 1, filters = 64, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 2, filters = 128, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 64, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 5, strides = 1, filters = 128, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 64, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 7, strides = 1, filters = 128, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 2, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 10, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 12, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 14, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 16, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 18, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 20, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 22, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 24, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 2, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 27, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 29, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 31, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 33, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 35, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 37, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 39, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 41, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 2, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 44, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 46, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 48, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = 50, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),

  # list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  # list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 255, kernel_size = 1, leaky_relu = FALSE, batch_normalization = FALSE),

  list(concatenate = 43, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),

  # list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  # list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 255, kernel_size = 1, leaky_relu = FALSE, batch_normalization = FALSE),

  list(concatenate = 26, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),

  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(concatenate = NULL, skip_connection = NULL, strides = 1, filters = 255, kernel_size = 1, leaky_relu = FALSE, batch_normalization = FALSE)
)

yolo3 <- function(input_shape) {
  input_img <- layer_input(shape = input_shape, name = 'input_img')

  blocks <- list()
  for (block in 1:length(yolo3_config)) {
    config <- yolo3_config[[block]]
    skip <- if (is.null(config$skip_connection)) NULL else blocks[[config$skip_connection]]
    concat <- if (is.null(config$concatenate)) NULL else blocks[[config$concatenate]]
    input <- if (block == 1) input_img else blocks[[block - 1]]
    blocks[[block]] <- yolo3_conv2d(input, id = block,
                                    skip_connection = skip,
                                    concatenate = concat,
                                    strides = config$strides, filters = config$filters,
                                    kernel_size = config$kernel_size, leaky_relu = config$leaky_relu,
                                    batch_normalization = config$batch_normalization)
  }

  output1 <- blocks[[57]] %>%
    yolo3_conv2d(id = "57_1",
                 skip_connection = NULL, concatenate = NULL,
                 strides = 1, filters = 1024, kernel_size = 3,
                 leaky_relu = TRUE, batch_normalization = TRUE) %>%
    yolo3_conv2d(id = "57_2",
                 skip_connection = NULL, concatenate = NULL,
                 strides = 1, filters = 255, kernel_size = 1,
                 leaky_relu = FALSE, batch_normalization = FALSE)
  output2 <- blocks[[63]] %>%
    yolo3_conv2d(id = "63_1",
                 skip_connection = NULL, concatenate = NULL,
                 strides = 1, filters = 512, kernel_size = 3,
                 leaky_relu = TRUE, batch_normalization = TRUE) %>%
    yolo3_conv2d(id = "63_2",
                 skip_connection = NULL, concatenate = NULL,
                 strides = 1, filters = 255, kernel_size = 1,
                 leaky_relu = FALSE, batch_normalization = FALSE)
  output3 <- blocks[[71]]
  outputs <- list(output1, output2, output3)

  keras_model(input_img, outputs)
}
