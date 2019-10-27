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
  grid_dims <- expand.grid(1:grid_h, 1:grid_w) %>% rename(w = Var2, h = Var1) %>%
    mutate(l = 1:(grid_w * grid_h), row = (l - 1) / grid_w , col = (l - 1) %% grid_w)
  pmap(grid_dims, function(w, h, l, row, col) {
    map2(boxes_coords, anchors, ~ {
      box_data <- preds[w, h, .x]
      anchor <- .y
      if (3>2) { # sigmoid(box_data[5]) > obj_threshold
        # Changing predictions to bbox center coordinates
        box_data[1] <- (sigmoid(box_data[1]) + col) / grid_w
        box_data[2] <- (sigmoid(box_data[2]) + row) / grid_h
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

correct_boxes <- function(boxes, image_w = 640, image_h = 386, net_w = 416, net_h = 416) {
  # if (net_w / image_w > net_h / image_h) {
  new_w <- net_w
  new_h <- net_h # (image_h * net_w) / image_w
  # } else {
  #   new_h <- net_w
  #   new_w <- (image_w * net_h) / image_h
  # }
  x_offset <- (net_w - new_w) / 2 / net_w
  x_scale <- new_w / net_w
  y_offset <- (net_h - new_h) / 2 / net_h
  y_scale <- new_h / net_h
  boxes %>% map(~ {
    image_boxes <- .x
    image_boxes %>% map(~ {
      data <- .x
      data[1] <- as.integer((data[1] - x_offset) / x_scale * image_w)
      data[2] <- as.integer((data[2] - y_offset) / x_scale * image_h)
      data[3] <- as.integer((data[3] - x_offset) / y_scale * image_w)
      data[4] <- as.integer((data[4] - y_offset) / y_scale * image_h)
      data
    })
  })
}

check_boxes_intersect <- function(box1, box2) {
  x_intersect <- box1[1] < box2[3] & box1[3] > box2[1]
  y_intersect <- box1[2] < box2[4] & box1[4] > box2[2]
  x_intersect & y_intersect
}

intersection_over_union <- function(box1, box2) {
  boxes_intersect <- check_boxes_intersect(box1, box2)
  intersection <- if (boxes_intersect) {
    (min(box1[3], box2[3]) - if (box2[1] < box1[1]) box1[1] else box2[1]) *
      (min(box1[4], box2[4]) - if (box2[2] < box1[2]) box1[2] else box2[2])
  } else {
    0
  }
  union <- (box1[3] - box1[1]) * (box1[4] - box1[2]) +
    (box2[3] - box2[1]) * (box2[4] - box2[2]) - intersection
  intersection / union
}

non_max_suppression <- function(boxes, n_class = 80, overlap_tresh = 0.5) {
  boxes %>% map(~ {
    images_boxes <- .x
    class_indexes <- 6:(n_class + 5)
    combinations_to_check <- class_indexes %>% map(~ {
      index <- .x
      images_boxes %>% keep(~ .x[index] == 1)
    }) %>% keep(~ length(.x) > 1)
    combinations_to_check %>% map(~ {
      current_boxes <- .x
      proba <- current_boxes %>% map_dbl(~ .x[5])
      combinations <- expand.grid(1:length(.x), 1:length(.x)) %>%
        rename(box1 = Var1, box2 = Var2)
      IoU <- combinations %>% pmap_dbl(function(box1, box2) {
        intersection_over_union(current_boxes[[box1]], current_boxes[[box2]])
      })
      unique_boxes <- combinations %>% bind_cols(IoU = IoU) %>%
        mutate(overlap = IoU >= overlap_tresh) %>%
        left_join(tibble(box1 = 1:length(current_boxes), proba = proba)) %>%
        group_by(box2, overlap) %>% mutate(proba_max = max(proba)) %>%
        filter(overlap == TRUE & proba == proba_max) %>% ungroup() %>%
        pull(box1) %>% unique()
      current_boxes[unique_boxes]
    })
  }) %>% unlist(recursive = FALSE)
}
