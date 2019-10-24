sigmoid <- function(x) 1 / (1 + exp(-x))

get_boxes <- function(preds, anchors, n_class = 80, n_box = 3, obj_threshold = 0.6, net_h = 416, net_w = 416) {
  n_images <- dim(preds[[1]])[1]
  1:n_images %>% map(~ {
    image_nr <- .x
    current_preds <- preds %>% map(~ .x[image_nr, , , ])
    map2(current_preds, anchors, ~ get_boxes_for_scale(preds = .x, anchors = .y, n_class, n_box,
                                                       class_threshold, net_w, net_h)) %>%
      unlist(recursive = FALSE)
  })
}

get_boxes_for_scale <- function(preds, anchors, n_class = 80, n_box = 3, obj_threshold = 0.6, net_w = 416, net_h = 416) {
  grid_w <- dim(preds)[1]
  grid_h <- dim(preds)[2]
  boxes_coords <- split(1:dim(preds)[3], rep(1:n_box, each = 5 + n_class))
  grid_dims <- expand.grid(1:grid_h, 1:grid_w) %>% rename(w = Var1, h = Var2)
  pmap(grid_dims, function(w, h) {
    map2(boxes_coords, anchors, ~ {
      box_data <- preds[w, h, .x]
      anchor <- .y
      if (sigmoid(box_data[5]) > obj_threshold) {
        # Changing predictions to bbox center coordinates
        box_data[1] <- (sigmoid(box_data[1]) + (w - 1) / grid_w) / grid_w
        box_data[2] <- (sigmoid(box_data[2]) + (h - 1) / grid_h) / grid_h
        box_data[3] <- anchor[1] * exp(box_data[3]) / net_w
        box_data[4] <- anchor[2] * exp(box_data[4]) / net_h
        box_data[5] <- sigmoid(box_data[5])
        box_data[6:length(box_data)] <- (box_data[5] * sigmoid(box_data[6:length(box_data)])) > obj_threshold
        # Changing centers to min/max coordinates
        xmin <- box_data[1] - box_data[3] / 2
        ymin <- box_data[2] - box_data[4] / 2
        xmax <- box_data[1] + box_data[3] / 2
        ymax <- box_data[2] + box_data[4] / 2
        box_data[1:4] <- c(xmin, ymin, xmax, ymax)
      } else {
        box_data <- NULL
      }
      box_data
    }) %>% keep(~ length(.x) > 1)
  }) %>% unlist(recursive = FALSE)
}
