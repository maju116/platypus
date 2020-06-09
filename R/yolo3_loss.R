transform_boxes_tf <- function(preds, anchors, n_class, net_h, net_w) {
  grid_size <- preds$get_shape()$as_list()[[2]]
  box_split <- tf$split(preds, num_or_size_splits = as.integer(c(2, 2, 1)), axis = as.integer(-1))
  box_xy <- k_sigmoid(box_split[[1]])
  box_wh <- box_split[[2]]
  score <- k_sigmoid(box_split[[3]])
  class_probs <- k_sigmoid(box_split[[4]])
  pred_box <- k_concatenate(list(box_xy, box_wh), axis = as.integer(-1))

  grid <- tf$meshgrid(tf$range(grid_size), tf$range(grid_size))
  grid <- tf$expand_dims(tf$stack(grid, axis = as.integer(-1)), axis = as.integer(2))

  box_xy = (box_xy + tf$cast(grid, tf$float32)) /  tf$cast(grid_size, tf$float32)
  box_wh = k_exp(box_wh) * anchors

  box_xmin_ymin = box_xy - box_wh / 2
  box_xmax_ymax = box_xy + box_wh / 2
  bbox = k_concatenate(list(box_xmin_ymin, box_xmax_ymax), axis = as.integer(-1))
  list(bbox, score, class_probs, pred_box)
}

yolo3_loss <- function(y_true, y_pred, anchors, n_class, nms_threshold = 0.6) {
  x
}
