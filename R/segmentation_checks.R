segmentation_input_mode = c("dir", "nested_dirs", "config_file", 1:3)

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

check_input_shape_u_net <- function(input_shape) {
  if (length(input_shape) != 3) {
    stop("'input_shape' must have 3 dimensions - image width, height and number of channels.")
  }
  if (any(input_shape < 1)) {
    stop("'input_shape' dimensions must be positive numbers.")
  }
  if (sum(as.integer(intToBits(input_shape[1]))) != 1 | sum(as.integer(intToBits(input_shape[2]))) != 1) {
    stop("Image width and height must be a power of two, e.g 64, 128, 256, ...")
  }
}

u_net_check <- function(input_shape, blocks, classes, filters, dropout, batch_normalization) {
  check_input_shape_u_net(input_shape)
  check_input_greater_than_min(blocks, min_value = 1)
  check_input_greater_than_min(classes, min_value = 2, "You need background and at least one other class.")
  check_input_greater_than_min(filters, min_value = 1)
  check_input_between(dropout, bounds = c(0, 1))
  check_input_in_set(batch_normalization, c(TRUE, FALSE))
}
