#' Sigmoid.
#' @description Applies sigmoid function.
#' @param x Input value - numeric.
#' @return Sigmoid of `x`.
#' @export
sigmoid <- function(x) 1 / (1 + exp(-x))

#' Logit.
#' @description Applies logit function.
#' @param x Input value - numeric from range `[0, 1]`.
#' @return Logit of `x`.
#' @export
logit <- function(x) log(x / (1 - x))

#' Transforms `Yolo3` predictions into valid boxes.
#' @description Transforms `Yolo3` predictions into valid boxes.
#' @param preds \code{\link[platypus]{yolo3}} model predictions.
#' @param anchors Prediction anchors. For exact format check \code{\link[platypus]{coco_anchors}}.
#' @param labels Character vector containing class labels. For example \code{\link[platypus]{coco_labels}}.
#' @param obj_threshold Minimum objectness score. Must be in range `[0, 1]`. All boxes with objectness score less than `obj_threshold` will be filtered out.
#' @param net_h Input layer height from trained \code{\link[platypus]{yolo3}} model. Must be divisible by `32`.
#' @param net_w Input layer width from trained \code{\link[platypus]{yolo3}} model. Must be divisible by `32`.
#' @param nms Logical. Should `Non-Maximum-Suppression` be applied.
#' @param nms_threshold `Non-Maximum-Suppression` threshold.
#' @param correct_hw Logical. Should height/width rescaling of bounding boxes be applied. If `TRUE` `xmin/xmax` coordinates are multiplied by `image_w` and `ymin/ymax` coordinates are multiplied by `image_h`.
#' @param image_h Rescaling factor for `ymin/ymax` box coordinates.
#' @param image_w Rescaling factor for `xmin/xmax` box coordinates.
#' @return List of `data.frames` containing bounding box coordinates and objectness/class scores.
#' @export
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

#' Transforms `Yolo3` predictions into valid box coordinates/scores.
#' @description Transforms `Yolo3` predictions into valid box coordinates/scores.
#' @param preds \code{\link[platypus]{yolo3}} model predictions.
#' @param anchors Prediction anchors. For exact format check \code{\link[platypus]{coco_anchors}}.
#' @param n_class Number of prediction classes.
#' @param anchors_per_grid Number of anchors/boxes per one output grid.
#' @param obj_threshold Minimum objectness score. Must be in range `[0, 1]`. All boxes with objectness score less than `obj_threshold` will be filtered out.
#' @param net_h Input layer height from trained \code{\link[platypus]{yolo3}} model. Must be divisible by `32`.
#' @param net_w Input layer width from trained \code{\link[platypus]{yolo3}} model. Must be divisible by `32`.
#' @return List of box coordinates/scores.
#' @export
transform_boxes <- function(preds, anchors, n_class, anchors_per_grid, obj_threshold, net_h, net_w) {
  n_images <- dim(preds[[1]])[1]
  1:n_images %>% map(~ {
    image_nr <- .x
    current_preds <- preds %>% map(~ .x[image_nr, , , , ])
    map2(current_preds, anchors, ~
           transform_boxes_for_grid(preds = .x, anchors = .y, n_class, anchors_per_grid,
                                     obj_threshold, net_h, net_w)) %>%
      unlist(recursive = FALSE)
  })
}

#' Transforms `Yolo3` predictions into valid box coordinates/scores.
#' @description Transforms `Yolo3` predictions into valid box coordinates/scores.
#' @importFrom purrr keep map2 pmap
#' @param preds \code{\link[platypus]{yolo3}} model predictions (from one grid).
#' @param anchors Prediction anchors (for one grid). For exact format check \code{\link[platypus]{coco_anchors}}.
#' @param n_class Number of prediction classes.
#' @param anchors_per_grid Number of anchors/boxes per one output grid.
#' @param obj_threshold Minimum objectness score. Must be in range `[0, 1]`. All boxes with objectness score less than `obj_threshold` will be filtered out.
#' @param net_h Input layer height from trained \code{\link[platypus]{yolo3}} model. Must be divisible by `32`.
#' @param net_w Input layer width from trained \code{\link[platypus]{yolo3}} model. Must be divisible by `32`.
#' @return List of box coordinates/scores.
transform_boxes_for_grid <- function(preds, anchors, n_class, anchors_per_grid, obj_threshold, net_h, net_w) {
  grid_h <- dim(preds)[1]
  grid_w <- dim(preds)[2]
  grid_dims <- expand.grid(1:grid_w, 1:grid_h) %>% select(h = Var2, w = Var1) %>%
    mutate(row = h - 1, col = w - 1)
  pmap(grid_dims, function(h, w, row, col) {
    map2(1:anchors_per_grid, anchors, ~ {
      box_data <- preds[h, w, .x, ]
      anchor <- .y
      if (sigmoid(box_data[5]) > obj_threshold) {
        box_data[1] <- (sigmoid(box_data[1]) + col) / grid_w
        box_data[2] <- (sigmoid(box_data[2]) + row) / grid_h
        box_data[3] <- anchor[1] * exp(box_data[3]) / net_w
        box_data[4] <- anchor[2] * exp(box_data[4]) / net_h
        box_data[5] <- sigmoid(box_data[5])
        box_data[6:length(box_data)] <- box_data[5] * sigmoid(box_data[6:length(box_data)])
        box_data[6:length(box_data)] <- (box_data[6:length(box_data)] == max(box_data[6:length(box_data)])) & (box_data[6:length(box_data)] > obj_threshold)
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

#' Checks if two bounding boxes intersect.
#' @description Checks if two bounding boxes intersect.
#' @param box1 Vector `(xmin, ymin, xmax, ymax)` with box coordinates.
#' @param box2 Vector `(xmin, ymin, xmax, ymax)` with box coordinates.
#' @return `TRUE` if `box1` and `box2` intersect, else `FALSE`.
check_boxes_intersect <- function(box1, box2) {
  x_intersect <- box1[1] < box2[3] & box1[3] > box2[1]
  y_intersect <- box1[2] < box2[4] & box1[4] > box2[2]
  x_intersect & y_intersect
}

#' Calculates `Intersection-Over-Union` for two bounding boxes.
#' @description Calculates `Intersection-Over-Union` for two bounding boxes.
#' @param box1 Vector `(xmin, ymin, xmax, ymax)` with box coordinates.
#' @param box2 Vector `(xmin, ymin, xmax, ymax)` with box coordinates.
#' @return `Intersection-Over-Union` for two bounding boxes.
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

#' Applies `Non-Maximum-Suppression` for a list of bounding boxes.
#' @description Applies `Non-Maximum-Suppression` for a list of bounding boxes.
#' @param boxes List of bounding boxes. Each box is a vector in format `(xmin, ymin, xmax, ymax)`.
#' @param n_class Number of prediction classes.
#' @param nms_threshold `Non-Maximum-Suppression` threshold.
#' @return List of non-overlapping bounding boxes.
#' @export
non_max_suppression <- function(boxes, n_class, nms_threshold) {
  boxes %>% map(~ {
    images_boxes <- .x
    class_indexes <- 6:(n_class + 5)
    combinations_to_check <- class_indexes %>% map(~ {
      index <- .x
      images_boxes %>% keep(~ .x[index] == 1)
    }) %>% keep(~ length(.x) >= 1)
    combinations_to_check %>% map(~ {
      current_boxes <- .x
      proba <- current_boxes %>% map_dbl(~ .x[5])
      combinations <- expand.grid(1:length(current_boxes), 1:length(current_boxes)) %>%
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

#' Transform list of bounding boxes into a `data.frame`.
#' @description Transform list of bounding boxes into a `data.frame`.
#' @param boxes List of bounding boxes.
#' @param labels Character vector containing class labels. For example \code{\link[platypus]{coco_labels}}.
#' @return List of `data.frames` containing bounding box coordinates and objectness/class scores.
#' @export
clean_boxes <- function(boxes, labels) {
  boxes %>% map(~ {
    boxes_data <- .x %>% map_df(~ as.data.frame(t(.x))) %>%
      set_names(c("xmin", "ymin", "xmax", "ymax", "p_obj", paste0("class", 1:length(labels))))
    boxes_data$label_id = apply(boxes_data %>% select(starts_with("class")), 1, which.max)
    boxes_data %>% select(-starts_with("class")) %>%
      mutate(label = labels[label_id])
  })
}

#' Rescales boxes.
#' @description Rescales boxes. `xmin/xmax` coordinates are multiplied by `image_w` and `ymin/ymax` coordinates are multiplied by `image_h`.
#' @param boxes `data.frame` with bounding boxes.
#' @param image_h Rescaling factor for `ymin/ymax` box coordinates.
#' @param image_w Rescaling factor for `xmin/xmax` box coordinates.
#' @return Rescaled bounding boxes.
#' @export
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
