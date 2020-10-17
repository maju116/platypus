segmentation_input_mode <- c("dir", "nested_dirs", "config_file", 1:3)

check_input_greater_than_min <- function(input, min_value = 1, message = NULL) {
  if (input < min_value) {
    stop(paste0("Argument '", deparse(substitute(input)), "' should be greater or equal to ",
                min_value, ". ", message))
  }
}

check_input_between <- function(input, bounds = c(0, 1), message = NULL) {
  if (input < bounds[1] | input > bounds[2]) {
    stop(paste0("Argument '", deparse(substitute(input)), "' should be between ", bounds[1], " and ",
                bounds[2], ". ", message))
  }
}

check_input_in_set <- function(input, set, message = NULL) {
  if (!(input %in% set)) {
    stop(paste0("Argument '", deparse(substitute(input)), "' should be equal to one of values: ",
                paste(set, collapse = ", "), ". ", message))
  }
}

check_input_shape_u_net <- function(net_h, net_w, grayscale) {
  if (any(c(net_h, net_w) < 1)) {
    stop("Dimensions must be positive numbers.")
  }
  if (sum(as.integer(intToBits(net_h))) != 1 | sum(as.integer(intToBits(net_w))) != 1) {
    stop("Image width and height must be a power of two, e.g 64, 128, 256, ...")
  }
  check_input_in_set(grayscale, c(TRUE, FALSE))
}

check_colormap <- function(colormap, n_class) {
  if (!is.list(colormap)) {
    stop("Colormap must be a list.")
  }
  if (length(colormap) != n_class) {
    stop("Colormap should have the same number of elements as number of classes.")
  }
  if (any(colormap %>% map(~ length(.)) %>% unlist() != 3)) {
    stop("Each element of colormap should have 3 values (RGB mapping).")
  }
}

u_net_check <- function(net_h, net_w, grayscale, blocks, n_class, filters, dropout, batch_normalization) {
  check_input_shape_u_net(net_h, net_w, grayscale)
  check_input_greater_than_min(blocks, min_value = 1)
  check_input_greater_than_min(n_class, min_value = 2, "You need background and at least one other class.")
  check_input_greater_than_min(filters, min_value = 1)
  check_input_between(dropout, bounds = c(0, 1))
  check_input_in_set(batch_normalization, c(TRUE, FALSE))
}

segmentation_generator_check <- function(colormap, mode, n_class, only_images, net_h, net_w, grayscale, shuffle) {
  check_colormap(colormap, n_class)
  check_input_in_set(mode, segmentation_input_mode)
  check_input_greater_than_min(n_class, min_value = 2, "You need background and at least one other class.")
  check_input_shape_u_net(net_h, net_w, grayscale)
  check_input_in_set(only_images, c(TRUE, FALSE))
  check_input_in_set(grayscale, c(TRUE, FALSE))
  check_input_in_set(shuffle, c(TRUE, FALSE))
}

yolo3_annot_formats <- c("pascal_voc", "labelme")

check_input_shape_yolo3 <- function(net_h, net_w, grayscale) {
  if (any(c(net_h, net_w) < 1)) {
    stop("Dimensions must be positive numbers.")
  }
  if (net_h%%32 != 0 | net_w%%32 != 0) {
    stop("Image width and height must be a divisible by 32, e.g 32, 64, 96, 128, ...")
  }
  check_input_in_set(grayscale, c(TRUE, FALSE))
}

check_anchors <- function(anchors) {
  if (!is.list(anchors)) {
    stop("Anchors must be a list.")
  }
  if (length(anchors) != 3) {
    stop("Anchors should have 3 elements (one for each grid).")
  }
  if (any(unlist(anchors, recursive = FALSE) %>% map(~ length(.)) %>% unlist() != 2)) {
    stop("Each anchor should have 2 values (height and width).")
  }
}

yolo3_check <- function(net_h, net_w, grayscale, n_class, anchors) {
  check_input_shape_yolo3(net_h, net_w, grayscale)
  check_input_greater_than_min(n_class, min_value = 1, "You need at least one class.")
  check_anchors(anchors)
}

yolo3_generator_check <- function(only_images, net_h, net_w, annot_format,
                                  grayscale, anchors, labels) {
  check_input_in_set(only_images, c(TRUE, FALSE))
  check_input_shape_yolo3(net_h, net_w, grayscale)
  check_input_in_set(annot_format, yolo3_annot_formats)
  check_anchors(anchors)
  check_input_greater_than_min(labels, min_value = 1)
}
