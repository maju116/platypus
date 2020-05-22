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
    when(!is.null(skip_connection) ~ layer_add(list(skip_connection, .), name = paste0("add_", id + 1)), ~ .) %>%
    when(!is.null(concatenate) ~ layer_upsampling_2d(., size = 2, name = paste0("upsample_", id + 1)), ~ .) %>%
    when(!is.null(concatenate) ~ layer_concatenate(list(concatenate, .), name = paste0("concat_", id + 2)), ~ .)
}

yolo3_config <- list(
  list(id = 0, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 32, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 1, concatenate = NULL, skip_connection = NULL, strides = 2, filters = 64, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 2, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 32, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 3, concatenate = NULL, skip_connection = 2, strides = 1, filters = 64, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 5, concatenate = NULL, skip_connection = NULL, strides = 2, filters = 128, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 6, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 64, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 7, concatenate = NULL, skip_connection = 5, strides = 1, filters = 128, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 9, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 64, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 10, concatenate = NULL, skip_connection = 7, strides = 1, filters = 128, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 12, concatenate = NULL, skip_connection = NULL, strides = 2, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 13, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 14, concatenate = NULL, skip_connection = 10, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 16, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 17, concatenate = NULL, skip_connection = 12, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 19, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 20, concatenate = NULL, skip_connection = 14, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 22, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 23, concatenate = NULL, skip_connection = 16, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 25, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 26, concatenate = NULL, skip_connection = 18, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 28, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 29, concatenate = NULL, skip_connection = 20, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 31, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 32, concatenate = NULL, skip_connection = 22, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 34, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 35, concatenate = NULL, skip_connection = 24, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 37, concatenate = NULL, skip_connection = NULL, strides = 2, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 38, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 39, concatenate = NULL, skip_connection = 27, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 41, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 42, concatenate = NULL, skip_connection = 29, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 44, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 45, concatenate = NULL, skip_connection = 31, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 47, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 48, concatenate = NULL, skip_connection = 33, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 50, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 51, concatenate = NULL, skip_connection = 35, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 53, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 54, concatenate = NULL, skip_connection = 37, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 56, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 57, concatenate = NULL, skip_connection = 39, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 59, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 60, concatenate = NULL, skip_connection = 41, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 62, concatenate = NULL, skip_connection = NULL, strides = 2, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 63, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 64, concatenate = NULL, skip_connection = 44, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 66, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 67, concatenate = NULL, skip_connection = 46, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 69, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 70, concatenate = NULL, skip_connection = 48, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 72, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 73, concatenate = NULL, skip_connection = 50, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 75, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 76, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 77, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 78, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 79, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),

  # list(id = 80, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 1024, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  # list(id = 81, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 255, kernel_size = 1, leaky_relu = FALSE, batch_normalization = FALSE),

  list(id = 84, concatenate = 43, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 87, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 88, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 89, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 90, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 91, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),

  # list(id = 92, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 512, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  # list(id = 93, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 255, kernel_size = 1, leaky_relu = FALSE, batch_normalization = FALSE),

  list(id = 96, concatenate = 26, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),

  list(id = 99, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 100, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 101, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 102, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 103, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 128, kernel_size = 1, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 104, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 256, kernel_size = 3, leaky_relu = TRUE, batch_normalization = TRUE),
  list(id = 105, concatenate = NULL, skip_connection = NULL, strides = 1, filters = 255, kernel_size = 1, leaky_relu = FALSE, batch_normalization = FALSE)
)

yolo3 <- function(input_shape) {
  input_img <- layer_input(shape = input_shape, name = 'input_img')

  blocks <- list()
  for (block in 1:length(yolo3_config)) {
    config <- yolo3_config[[block]]
    skip <- if (is.null(config$skip_connection)) NULL else blocks[[config$skip_connection]]
    concat <- if (is.null(config$concatenate)) NULL else blocks[[config$concatenate]]
    input <- if (block == 1) input_img else blocks[[block - 1]]
    blocks[[block]] <- yolo3_conv2d(input, id = config$id,
                                    skip_connection = skip,
                                    concatenate = concat,
                                    strides = config$strides, filters = config$filters,
                                    kernel_size = config$kernel_size, leaky_relu = config$leaky_relu,
                                    batch_normalization = config$batch_normalization)
  }

  output1 <- blocks[[57]] %>%
    yolo3_conv2d(id = 80,
                 skip_connection = NULL, concatenate = NULL,
                 strides = 1, filters = 1024, kernel_size = 3,
                 leaky_relu = TRUE, batch_normalization = TRUE) %>%
    yolo3_conv2d(id = 81,
                 skip_connection = NULL, concatenate = NULL,
                 strides = 1, filters = 255, kernel_size = 1,
                 leaky_relu = FALSE, batch_normalization = FALSE)
  output2 <- blocks[[63]] %>%
    yolo3_conv2d(id = 92,
                 skip_connection = NULL, concatenate = NULL,
                 strides = 1, filters = 512, kernel_size = 3,
                 leaky_relu = TRUE, batch_normalization = TRUE) %>%
    yolo3_conv2d(id = 93,
                 skip_connection = NULL, concatenate = NULL,
                 strides = 1, filters = 255, kernel_size = 1,
                 leaky_relu = FALSE, batch_normalization = FALSE)
  output3 <- blocks[[71]]
  outputs <- list(output1, output2, output3)

  keras_model(input_img, outputs)
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
