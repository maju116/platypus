get_boxes <- function(preds, n_class = 80, n_box = 3, class_threshold = 0.6) {
  grid_w <- dim(preds)[1]
  grid_h <- dim(preds)[2]
  boxes_coords <- split(1:dim(preds)[3], rep(1:n_box, each = 5 + n_class))
  anchors <- list(c(116, 90), c(156, 198), c(373, 326))
  grid_dims <- expand.grid(1:grid_h, 1:grid_w) %>% rename(w = Var1, h = Var2)
  pmap(grid_dims, function(w, h) {
    map2(boxes_coords, anchors, ~ {
      box_data <- preds[w, h, .x]
      anchor <- .y
      # Changing predictions to bbox center coordinates
      box_data[1] <- (sigmoid(box_data[1]) + (w - 1) / grid_w) / grid_w
      box_data[2] <- (sigmoid(box_data[2]) + (h - 1) / grid_h) / grid_h
      box_data[3] <- anchor[1] * exp(box_data[3]) / net_w
      box_data[4] <- anchor[2] * exp(box_data[4]) / net_h
      box_data[5] <- sigmoid(box_data[5])
      box_data[6:length(box_data)] <- (box_data[5] * sigmoid(box_data[6:length(box_data)])) > class_threshold
      # Changing centers to min/max coordinates
      xmin <- box_data[1] - box_data[3] / 2
      ymin <- box_data[2] - box_data[4] / 2
      xmax <- box_data[1] + box_data[3] / 2
      ymax <- box_data[2] + box_data[4] / 2
      box_data[1:4] <- c(xmin, ymin, xmax, ymax)
      box_data
    })
  }) %>% unlist(recursive = FALSE)
}
