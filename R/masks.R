# Turning masks into things you can look at.
#
# This is the half of the package that has nothing to do with training. It exists because
# the useful question is rarely "what is the Dice" - it is "where is the model wrong, and
# does that matter for what I am measuring". A number cannot answer that and a picture can.
#
# Everything here works on ordinary R arrays and needs no Python, so it is equally useful
# on masks that came from somewhere else entirely.

#' Paint a mask with its colormap
#'
#' @param mask Class indices, counted from 1: a `height x width` matrix, or an
#'   `image x height x width` array for several at once.
#' @param colormap A list of RGB triples, one per class; see [binary_colormap].
#' @return The same shape with a trailing RGB axis, values 0-255.
#' @export
#' @examples
#' mask <- matrix(c(1, 1, 2, 2), nrow = 2)
#' dim(mask_colours(mask, binary_colormap))
mask_colours <- function(mask, colormap) {
  palette <- do.call(rbind, lapply(colormap, as.integer))
  check_classes(mask, nrow(palette))
  coloured <- palette[as.vector(mask), , drop = FALSE]
  array(as.integer(coloured), dim = c(dim(mask), 3L))
}

#' Read a colour mask back to class indices
#'
#' The inverse of [mask_colours()]. Anything matching no colour becomes background, and
#' `coverage` in the result says how much of the mask that was - a number near 1 means the
#' colormap does not describe this dataset, which is the quietest way to train on nothing.
#'
#' @param mask An RGB array: `height x width x 3`, or with a leading image axis.
#' @param colormap A list of RGB triples, one per class.
#' @param tolerance Allow this much difference per channel. Useful after lossy resizing,
#'   though masks should be resized with `nearest = TRUE` so it is not needed.
#' @return A list with `classes` (indices counted from 1) and `coverage` (the fraction of
#'   pixels that matched a colour).
#' @export
#' @examples
#' coloured <- mask_colours(matrix(c(1, 2, 2, 1), nrow = 2), binary_colormap)
#' mask_classes(coloured, binary_colormap)$classes
mask_classes <- function(mask, colormap, tolerance = 0) {
  dims <- dim(mask)
  if (utils::tail(dims, 1) < 3L) {
    stop("expected an RGB mask with three channels last, got ",
         paste(dims, collapse = "x"), call. = FALSE)
  }
  spatial <- dims[-length(dims)]
  flat <- matrix(as.numeric(mask), ncol = dims[length(dims)])[, 1:3, drop = FALSE]

  classes <- integer(nrow(flat))
  matched <- logical(nrow(flat))
  # Later colours win, so an explicit class beats the background it overlaps.
  for (index in seq_along(colormap)) {
    target <- as.numeric(colormap[[index]])
    hit <- abs(flat[, 1] - target[1]) <= tolerance &
           abs(flat[, 2] - target[2]) <= tolerance &
           abs(flat[, 3] - target[3]) <= tolerance
    classes[hit] <- index
    matched <- matched | hit
  }
  classes[classes == 0L] <- 1L

  list(classes = array(classes, dim = spatial), coverage = mean(matched))
}

#' Read mask files as class indices
#'
#' @inheritParams read_images
#' @param colormap A list of RGB triples, one per class.
#' @param tolerance Passed to [mask_classes()].
#' @return An array of class indices, counted from 1.
#' @export
read_masks <- function(paths, colormap, size = NULL, tolerance = 0) {
  # Nearest, always. Interpolating a mask invents colours belonging to no class, which
  # then quietly become background - and the plot would show a disagreement that is
  # nothing but the resizing.
  coloured <- read_images(paths, size = size, channels = 3, nearest = TRUE)
  mask_classes(coloured, colormap, tolerance = tolerance)$classes
}

#' Lay a mask over the image it belongs to
#'
#' The picture that answers "where is it looking". Background - class 1 - is left alone,
#' so only the classes that were found are tinted.
#'
#' @param image A `height x width x channels` array. Values in 0-1 or 0-255; greyscale is
#'   repeated across three channels.
#' @param mask Class indices, counted from 1, matching the image's height and width.
#' @param colormap A list of RGB triples, one per class.
#' @param alpha How strongly to tint, 0 to 1.
#' @return A `height x width x 3` array of 0-255 integers.
#' @export
overlay_mask <- function(image, mask, colormap, alpha = 0.55) {
  image <- as_rgb(image)
  if (!identical(dim(image)[1:2], dim(mask)[1:2])) {
    stop("image is ", paste(dim(image)[1:2], collapse = "x"), " but mask is ",
         paste(dim(mask)[1:2], collapse = "x"), "; they must match.", call. = FALSE)
  }
  tint <- mask_colours(mask, colormap)
  foreground <- as.vector(mask) > 1L
  blended <- image
  for (channel in 1:3) {
    layer <- blended[, , channel]
    layer[foreground] <- (1 - alpha) * layer[foreground] +
      alpha * tint[, , channel][foreground]
    blended[, , channel] <- layer
  }
  array(as.integer(round(blended)), dim = dim(blended))
}

#' Where a prediction and the truth disagree
#'
#' Three colours rather than one: what was found, what was missed, and what was invented.
#' A Dice of 0.85 says nothing about which of the three is costing you, and for most
#' clinical questions they are not interchangeable - a missed lesion and a false alarm
#' are different kinds of wrong.
#'
#' @param image A `height x width x channels` array.
#' @param prediction,truth Class indices, counted from 1.
#' @param alpha How strongly to tint.
#' @param colours Named colours for `hit`, `missed` and `false_alarm`.
#' @return A `height x width x 3` array of 0-255 integers.
#' @export
overlay_agreement <- function(image, prediction, truth, alpha = 0.55,
                              colours = c(hit = "#3CDC5A", missed = "#E63C3C",
                                          false_alarm = "#F0C83C")) {
  image <- as_rgb(image)
  predicted <- as.vector(prediction) > 1L
  actual <- as.vector(truth) > 1L
  parts <- list(hit = predicted & actual,
                missed = !predicted & actual,
                false_alarm = predicted & !actual)

  blended <- image
  for (part in names(parts)) {
    where <- parts[[part]]
    if (!any(where)) next
    rgb <- grDevices::col2rgb(colours[[part]])[, 1]
    for (channel in 1:3) {
      layer <- blended[, , channel]
      layer[where] <- (1 - alpha) * layer[where] + alpha * rgb[[channel]]
      blended[, , channel] <- layer
    }
  }
  array(as.integer(round(blended)), dim = dim(blended))
}

#' Combine several binary masks into one
#'
#' The Data Science Bowl stores one file per nucleus; datasets like it are common. Later
#' masks win where they overlap, so an explicit class beats the background it sits on.
#'
#' @param masks A list of `height x width` matrices of class indices.
#' @return One `height x width` matrix.
#' @export
unite_masks <- function(masks) {
  if (!length(masks)) stop("`masks` is empty.", call. = FALSE)
  shapes <- unique(lapply(masks, dim))
  if (length(shapes) > 1L) {
    stop("all masks must be the same size; found ",
         paste(vapply(shapes, paste, character(1), collapse = "x"), collapse = " and "),
         call. = FALSE)
  }
  Reduce(function(a, b) pmax(a, b), masks)
}

#' How much of each class a mask covers
#'
#' A quick sanity check before training: a foreground of half a percent is a problem the
#' loss has to be chosen for, and it is better to know before the run than after.
#'
#' @param mask Class indices, counted from 1.
#' @param labels Optional names for the classes.
#' @return A data frame with one row per class.
#' @export
mask_coverage <- function(mask, labels = NULL) {
  counts <- tabulate(as.vector(mask))
  classes <- seq_along(counts)
  data.frame(
    class = classes,
    label = if (is.null(labels)) paste("class", classes) else labels[classes],
    pixels = counts,
    fraction = counts / length(mask),
    stringsAsFactors = FALSE
  )
}

#' Any array to a 0-255 RGB array
#' @keywords internal
#' @noRd
as_rgb <- function(image) {
  if (length(dim(image)) == 2L) image <- array(image, dim = c(dim(image), 1L))
  if (max(image, na.rm = TRUE) <= 1 + 1e-8) image <- image * 255
  channels <- dim(image)[3]
  if (channels == 1L) {
    image <- array(rep(image, 3L), dim = c(dim(image)[1:2], 3L))
  } else if (channels > 3L) {
    image <- image[, , 1:3, drop = FALSE]
  }
  image
}

#' @keywords internal
#' @noRd
check_classes <- function(mask, n_class) {
  values <- unique(as.vector(mask))
  bad <- values[values < 1L | values > n_class]
  if (length(bad)) {
    stop("mask contains class ", bad[[1]], " but the colormap defines ", n_class,
         " classes. Classes are counted from 1.", call. = FALSE)
  }
  invisible(TRUE)
}
