darknet53_conv2d <- function(input, strides, filters, kernel_size,
                             batch_normalization = TRUE, leaky_relu = TRUE, id) {
  input %>%
    when(strides > 1 ~ layer_zero_padding_2d(., list(c(1, 0), c(1, 0)), name = paste0("zero_pad_", id)), ~ .) %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size, strides = strides,
                  padding = if (strides > 1) "valid" else "same",
                  name = paste0("conv2d_", id),
                  use_bias = if (batch_normalization) FALSE else TRUE,
                  kernel_initializer = initializer_glorot_uniform(),
                  kernel_regularizer = regularizer_l2(l = 5e-4)) %>%
    when(batch_normalization ~ layer_batch_normalization(., name = paste0("batch_norm_", id),
                                                         center = TRUE, scale = TRUE,
                                                         momentum = 0.99, epsilon = 1e-3), ~ .) %>%
    when(leaky_relu ~ layer_activation_leaky_relu(., alpha = 0.1, name = paste0("leaky_relu_", id)), ~ .)
}

darknet53_residual_block <- function(input, id, filters, blocks) {
  id_ <- id
  net_out <- input %>%
    darknet53_conv2d(strides = 2, filters = filters, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE, id = id_)
  id_ <- id_ + 1
  for (block in 1:blocks) {
    add_layer <- net_out
    net_out <- net_out %>%
      darknet53_conv2d(strides = 1, filters = round(filters / 2), kernel_size = 1,
                       batch_normalization = TRUE, leaky_relu = TRUE, id = id_)
    id_ <- id_ + 1
    net_out <- net_out %>%
      darknet53_conv2d(strides = 1, filters = filters, kernel_size = 3,
                       batch_normalization = TRUE, leaky_relu = TRUE, id = id_)
    id_ <- id_ + 1
    net_out <- layer_add(list(add_layer, net_out), name = paste0("add_", id_ + 1))
  }
  list(net_out = net_out, id_ = id_)
}

darknet53 <- function(input) {
  id_ <- 1
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = 32, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE, id = id_)
  id_ <- id_ + 1
  net_out <- net_out %>% darknet53_residual_block(id = id_, filters = 64, blocks = 1)
  net_out <- net_out$net_out %>% darknet53_residual_block(id = net_out$id_, filters = 128, blocks = 2)
  net_out <- net_out$net_out %>% darknet53_residual_block(id = net_out$id_, filters = 256, blocks = 8)
  out1 <- net_out$net_out
  net_out <- net_out$net_out %>% darknet53_residual_block(id = net_out$id_, filters = 512, blocks = 8)
  out2 <- net_out$net_out
  net_out <- net_out$net_out %>% darknet53_residual_block(id = net_out$id_, filters = 1024, blocks = 4)
  # list(out1 = out1, out2 = out2, out3 = net_out$net_out, id_ = net_out$id_)
  keras_model(input, list(out1, out2, net_out$net_out))
}

yolo_block <- function(input, id, filters, out_filters) {
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = filters, kernel_size = 1,
                     batch_normalization = TRUE, leaky_relu = TRUE, id = id)
  id_ <- id + 1
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = filters * 2, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE, id = id_)
  id_ <- id_ + 1
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = filters, kernel_size = 1,
                     batch_normalization = TRUE, leaky_relu = TRUE, id = id_)
  id_ <- id_ + 1
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = filters * 2, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE, id = id_)
  id_ <- id_ + 1
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = filters, kernel_size = 1,
                     batch_normalization = TRUE, leaky_relu = TRUE, id = id_)
  id_ <- id_ + 1
  out1 <- net_out
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = filters * 2, kernel_size = 3,
                     batch_normalization = TRUE, leaky_relu = TRUE, id = id_)
  id_ <- id_ + 1
  net_out <- input %>%
    darknet53_conv2d(strides = 1, filters = out_filters, kernel_size = 1,
                     batch_normalization = FALSE, leaky_relu = FALSE, id = id_)
  id_ <- id_ + 1
  list(out1 = out1, net_out = net_out, id_ = id_)
}

yolo3 <- function(input_shape) {
  input_img <- layer_input(shape = input_shape, name = 'input_img')
  darknet <- darknet53(input_img)
  num_anchors <- 3
  num_classes <- 80
  yolo_block1 <- yolo_block(darknet$out3, darknet$id_, 512, num_anchors * (num_classes + 5))
  conv2d_60 <- yolo_block1$out1 %>%
    darknet53_conv2d(strides = 1, filters = 256, kernel_size = 1,
                     batch_normalization = FALSE, leaky_relu = FALSE, id = yolo_block1$id_)
  upsample_0 <- conv2d_60 %>% layer_upsampling_2d(size = 2, name = paste0("upsample_", yolo_block1$id_))
  route_0 <- layer_concatenate(list(darknet$out2, upsample_0), name = paste0("concat_", yolo_block1$id_))
  id_ <- yolo_block1$id_ + 1
  yolo_block2 <- yolo_block(route_0, id_, 256, num_anchors * (num_classes + 5))
  conv2d_68 <- yolo_block2$out1 %>%
    darknet53_conv2d(strides = 1, filters = 128, kernel_size = 1,
                     batch_normalization = FALSE, leaky_relu = FALSE, id = yolo_block2$id_)
  upsample_1 <- conv2d_68 %>% layer_upsampling_2d(size = 2, name = paste0("upsample_", id_))
  route_1 <- layer_concatenate(list(darknet$out1, upsample_1), name = paste0("concat_", id_))
  id_ <- yolo_block2$id_ + 1
  yolo_block3 <- yolo_block(route_1, id_, 128, num_anchors * (num_classes + 5))
  outputs <- list(yolo_block1$net_out, yolo_block2$net_out, yolo_block3$net_out)
  keras_model(input_img, outputs)
}
