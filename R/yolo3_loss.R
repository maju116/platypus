transform_boxes_tf <- function(preds, anchors, n_class, net_h, net_w) {
  grid_h <- preds$get_shape()$as_list()[[2]]
  grid_w <- preds$get_shape()$as_list()[[3]]
  box_split <- tf$split(preds, num_or_size_splits = as.integer(c(1, 1, 1, 1, 1, n_class)), axis = as.integer(-1))
  box_x <- k_sigmoid(box_split[[1]])
  box_y <- k_sigmoid(box_split[[2]])
  box_w <- box_split[[3]]
  box_h <- box_split[[4]]
  score <- k_sigmoid(box_split[[5]])
  class_probs <- k_sigmoid(box_split[[6]])
  pred_box <- k_concatenate(list(box_x, box_y, box_w, box_h), axis = as.integer(-1))

  grid <- tf$meshgrid(tf$range(grid_w), tf$range(grid_h))
  grid_col <- tf$expand_dims(tf$expand_dims(grid[[1]], axis = as.integer(-1)), axis = as.integer(-1))
  grid_row <- tf$expand_dims(tf$expand_dims(grid[[2]], axis = as.integer(-1)), axis = as.integer(-1))

  box_x = (box_x + tf$cast(grid_col, tf$float32)) /  tf$cast(grid_w, tf$float32)
  box_y = (box_y + tf$cast(grid_row, tf$float32)) /  tf$cast(grid_h, tf$float32)

  anchors_tf <- tf$constant(anchors, tf$float32)
  anchors_tf <- tf$expand_dims(tf$expand_dims(anchors_tf, axis = as.integer(0)), axis = as.integer(0))
  anchors_tf <- tf$split(anchors_tf, num_or_size_splits = as.integer(c(1, 1)), axis = as.integer(-1))
  box_w <- k_exp(box_w) * anchors_tf[[1]] / tf$cast(net_w, tf$float32)
  box_h <- k_exp(box_h) * anchors_tf[[2]] / tf$cast(net_h, tf$float32)

  box_xmin <- box_x - box_w / 2
  box_ymin <- box_y - box_h / 2
  box_xmax <- box_x + box_w / 2
  box_ymax <- box_y + box_h / 2
  bbox <- k_concatenate(list(box_xmin, box_ymin, box_xmax, box_ymax), axis = as.integer(-1))
  # list(bbox, score, class_probs, pred_box)
  k_concatenate(list(bbox, score, class_probs), axis = as.integer(-1))
}

yolo3_loss <- function(y_true, y_pred, anchors, n_class, nms_threshold = 0.6) {
  x
}
