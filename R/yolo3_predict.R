sigmoid <- function(x) 1 / (1 + exp(-x))

get_boxes <- function(preds, anchors, labels, obj_threshold = 0.6,
                      net_h = 416, net_w = 416, nms = TRUE,
                      nms_threshold = 0.6, correct_hw = FALSE,
                      image_h = NULL, image_w = NULL) {
  n_class = length(labels)
  anchors_per_grid = length(anchors[[1]])
  preds %>%
    transform_boxes(anchors, n_class, anchors_per_grid, obj_threshold, net_h, net_w) %>%
    when(nms ~ non_max_suppression(., n_class, nms_threshold), ~ .) %>%
    clean_boxes(labels) %>%
    when(correct_hw ~ correct_boxes(., image_h, image_w), ~ .)
}

transform_boxes <- function(preds, anchors, n_class, anchors_per_grid, obj_threshold, net_h, net_w) {
  n_images <- dim(preds[[1]])[1]
  1:n_images %>% map(~ {
    image_nr <- .x
    current_preds <- preds %>% map(~ .x[image_nr, , , , ])
    map2(current_preds, anchors, ~
           transform_boxes_for_image(preds = .x, anchors = .y, n_class, anchors_per_grid,
                                     obj_threshold, net_h, net_w)) %>%
      unlist(recursive = FALSE)
  })
}

transform_boxes_for_image <- function(preds, anchors, n_class, anchors_per_grid, obj_threshold, net_h, net_w) {
  grid_h <- dim(preds)[1]
  grid_w <- dim(preds)[2]
  grid_dims <- expand.grid(1:grid_h, 1:grid_w) %>% select(h = Var2, w = Var1) %>%
    mutate(l = 1:(grid_h * grid_w), row = floor((l - 1) / grid_h) , col = (l - 1) %% grid_w)
  pmap(grid_dims, function(h, w, l, row, col) {
    map2(1:anchors_per_grid, anchors, ~ {
      box_data <- preds[h, w, .x, ]
      anchor <- .y
      if (sigmoid(box_data[5]) > obj_threshold) {
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

non_max_suppression <- function(boxes, n_class, nms_threshold = 0.5) {
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
        mutate(overlap = IoU >= nms_threshold) %>%
        left_join(tibble(box1 = 1:length(current_boxes), proba = proba), by = "box1") %>%
        group_by(box2, overlap) %>% mutate(proba_max = max(proba)) %>%
        filter(overlap == TRUE & proba == proba_max) %>% ungroup() %>%
        pull(box1) %>% unique()
      current_boxes[unique_boxes]
    }) %>% unlist(recursive = FALSE)
  })
}

clean_boxes <- function(boxes, labels) {
  boxes %>% map(~ {
    boxes_data <- .x %>% map_df(~ as.data.frame(t(.x))) %>%
      set_names(c("xmin", "ymin", "xmax", "ymax", "p_obj", paste0("class", 1:length(labels))))
    boxes_data$label_id = apply(boxes_data %>% select(starts_with("class")), 1, which.max)
    boxes_data %>% select(-starts_with("class")) %>%
      mutate(label = labels[label_id])
  })
}

correct_boxes <- function(boxes, image_h, image_w) {
  boxes %>% map(~ {
    current_boxes <- .x
    current_boxes %>%
      mutate(
        xmin = as.integer(xmin * image_w),
        ymin = as.integer(ymin * image_h),
        xmax = as.integer(xmax * image_w),
        ymax = as.integer(ymax * image_h)
      )
  })
}
