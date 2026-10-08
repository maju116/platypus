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
#' dim(classes_to_colours(mask, binary_colormap))
classes_to_colours <- function(mask, colormap) {
  palette <- do.call(rbind, lapply(colormap, as.integer))
  check_classes(mask, nrow(palette))
  coloured <- palette[as.vector(mask), , drop = FALSE]
  array(as.integer(coloured), dim = c(dim(mask), 3L))
}

#' Read a colour mask back to class indices
#'
#' The inverse of [classes_to_colours()]. Anything matching no colour becomes background, and
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
#' coloured <- classes_to_colours(matrix(c(1, 2, 2, 1), nrow = 2), binary_colormap)
#' colours_to_classes(coloured, binary_colormap)$classes
colours_to_classes <- function(mask, colormap, tolerance = 0) {
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
    # Three parallel comparisons, aligned so that being the same shape is visible.
    hit <- abs(flat[, 1] - target[1]) <= tolerance &
           abs(flat[, 2] - target[2]) <= tolerance &  # nolint: indentation_linter.
           abs(flat[, 3] - target[3]) <= tolerance
    classes[hit] <- index
    matched <- matched | hit
  }
  classes[classes == 0L] <- 1L

  list(classes = array(classes, dim = spatial), coverage = mean(matched))
}

#' Read mask files as class indices
#'
#' Masks come two ways, and this reads both. `colormap` is for masks stored as pictures, one
#' colour per class. `labels` is for masks stored as label maps - a vector of the voxel values,
#' in class order - which is how volume formats store them.
#'
#' @inheritParams read_images
#' @param colormap A list of RGB triples, one per class. Give this or `labels`.
#' @param labels The voxel value of each class, in class order: `c(0, 1)` for a binary
#'   segmentation. For NIfTI and other label maps. Give this or `colormap`.
#' @param tolerance Passed to [colours_to_classes()]. Ignored for label maps, which are compared to
#'   within half a unit - a label map that has been resampled or merely passed through a float
#'   will not satisfy an exact comparison, and a label that fails to match becomes background.
#' @return An array of class indices, counted from 1.
#' @export
#' @examples
#' \dontrun{
#' # Masks stored as pictures: the colormap says which colour is which class.
#' truth <- read_masks(list.files("masks/", full.names = TRUE),
#'                     colormap = list(c(0, 0, 0), c(255, 255, 255)))
#'
#' # Masks stored as label maps, which is how volumes do it. Exactly one of the two.
#' truth <- read_masks("case_01/seg.nii.gz", labels = c(0, 1))
#'
#' # `tolerance` is for masks that have been through a lossy resize, where a colour
#' # that should be (255, 0, 0) arrives as (254, 1, 0) and matches nothing.
#' truth <- read_masks(paths, colormap = voc_colormap, tolerance = 2)
#' }
read_masks <- function(paths, colormap = NULL, labels = NULL, size = NULL, tolerance = 0) {
  if (is.null(colormap) == is.null(labels)) {
    stop("give exactly one of `colormap` (masks stored as pictures) and `labels` ",
         "(masks stored as label maps, which is how volumes do it).", call. = FALSE)
  }

  # Nearest, always. Interpolating a mask invents values belonging to no class, which then
  # quietly become background - and the plot would show a disagreement that is nothing but the
  # resizing.
  if (!is.null(labels)) {
    values <- read_images(paths, size = size, channels = 1, nearest = TRUE)
    return(label_classes(values, labels))
  }
  coloured <- read_images(paths, size = size, channels = 3, nearest = TRUE)
  colours_to_classes(coloured, colormap, tolerance = tolerance)$classes
}

#' Label values to class indices
#'
#' @param values An array whose last axis is a single channel, as [read_images()] returns for
#'   `channels = 1`.
#' @param labels The voxel value of each class, in class order.
#' @return An array of class indices counted from 1, with the channel axis dropped.
#' @keywords internal
#' @noRd
label_classes <- function(values, labels) {
  dims <- dim(values)
  spatial <- dims[-length(dims)]
  flat <- as.numeric(values)

  classes <- integer(length(flat))
  for (index in seq_along(labels)) {
    # Half a unit, because the values arrive as floats: exact comparison would miss a label
    # map that has been resampled, and a label that misses becomes background.
    classes[abs(flat - as.numeric(labels[[index]])) <= 0.5] <- index
  }
  classes[classes == 0L] <- 1L
  array(classes, dim = spatial)
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
#' @examples
#' image <- array(runif(8 * 8 * 3, 0.2, 0.6), dim = c(8, 8, 3))
#' mask <- matrix(1L, 8, 8); mask[2:4, 2:4] <- 2L
#' tinted <- overlay_mask(image, mask, colormap = list(c(0, 0, 0), c(255, 0, 0)))
#' dim(tinted)
overlay_mask <- function(image, mask, colormap, alpha = drawing_style()$overlay_alpha) {
  image <- as_rgb(image)
  if (!identical(dim(image)[1:2], dim(mask)[1:2])) {
    stop("image is ", paste(dim(image)[1:2], collapse = "x"), " but mask is ",
         paste(dim(mask)[1:2], collapse = "x"), "; they must match.", call. = FALSE)
  }
  tint <- classes_to_colours(mask, colormap)
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
#' @examples
#' image <- array(runif(8 * 8 * 3, 0.2, 0.6), dim = c(8, 8, 3))
#' truth <- matrix(1L, 8, 8); truth[2:4, 2:4] <- 2L
#' predicted <- matrix(1L, 8, 8); predicted[3:5, 2:4] <- 2L
#' shown <- overlay_agreement(image, prediction = predicted, truth = truth)
#' dim(shown)
#'
#' # Three colours, not one: a missed lesion and a false alarm cost different things,
#' # and a single overlap score hides which one you have.
overlay_agreement <- function(image, prediction, truth,
                              alpha = drawing_style()$overlay_alpha,
                              colours = drawing_style()$agreement_colours) {
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
#' @examples
#' # Data Science Bowl stores one file per nucleus, so a sample's mask arrives as several
#' # matrices that have to become one. Later masks win where they overlap.
#' one <- matrix(1L, 8, 8); one[2:4, 2:4] <- 2L
#' two <- matrix(1L, 8, 8); two[6:7, 5:7] <- 2L
#' united <- unite_masks(list(one, two))
#' table(united)
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
#' @examples
#' mask <- matrix(1L, 8, 8); mask[2:4, 2:4] <- 2L
#' mask_coverage(mask, labels = c("background", "nucleus"))
#'
#' # A class with 0 pixels is the thing to look for: its target channel is zero
#' # everywhere, so there is no gradient towards it and the model is never shown
#' # what it is being asked to find.
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

#' Write masks to files
#'
#' Predictions are only half useful while they are an array in a session. This writes them
#' as ordinary PNGs that open in anything - ImageJ, QuPath, a viewer, a colleague's
#' machine - and returns the paths, so it can sit at the end of a pipeline.
#'
#' Files go where you say, not beside the images they came from. Writing into somebody's
#' dataset directory is a surprising thing for a function to do, and a second run against
#' a different model would quietly mix its results in with the first.
#'
#' @param masks Class indices from [predict.platypus_fit()] - an `image x height x width`
#'   array, or one `height x width` matrix. Probabilities are accepted too and reduced to
#'   the most likely class.
#'
#'   For **volumes** use [save_volumes()], which writes NIfTI carrying the geometry of the
#'   scan each mask came from. This function cannot reliably tell a stack of volumes from a
#'   stack of images with a class axis - the ranks are identical - so it does not try; it
#'   refuses only the case a shape can prove.
#' @param dir Directory to write into; created if it does not exist.
#' @param colormap A list of RGB triples, one per class.
#' @param names File names, or the source image paths to take names from. Defaults to
#'   `mask_0001.png` and so on.
#' @param suffix Added before the extension, so predictions from different models can sit
#'   in one directory without overwriting each other.
#' @return The paths written, invisibly.
#' @export
#' @examples
#' \dontrun{
#' masks <- predict(fit, "unet", split = "test")
#' save_masks(masks, "predictions", binary_colormap, suffix = "_unet")
#' }
save_masks <- function(masks, dir, colormap, names = NULL, suffix = "") {
  if (length(dim(masks)) == 5L) {
    # The one case a shape can prove. A stack of volumes, (n, d, h, w), has the same rank as
    # a stack of images with a class axis, so this function cannot detect that one - which
    # is why the documentation points at save_volumes() rather than this check pretending to.
    stop("these are volumes (", paste(dim(masks), collapse = " x "), "). Use ",
         "`save_volumes()`, which writes NIfTI carrying the geometry of the scan each mask ",
         "came from - a volume mask without that cannot be laid over its scan by anything.",
         call. = FALSE)
  }
  if (length(dim(masks)) == 4L) {
    # Probabilities rather than classes: take the most likely, as predict() would.
    masks <- apply(masks, seq_len(length(dim(masks)) - 1L), which.max)
  }
  if (length(dim(masks)) == 2L) masks <- array(masks, dim = c(1L, dim(masks)))

  count <- dim(masks)[1]
  names <- if (is.null(names)) {
    sprintf("mask_%04d", seq_len(count))
  } else {
    tools::file_path_sans_ext(basename(as.character(names)))
  }
  if (length(names) != count) {
    stop("`names` has ", length(names), " entries but there are ", count, " masks.",
         call. = FALSE)
  }

  coloured <- lapply(seq_len(count), function(i) classes_to_colours(masks[i, , ], colormap))
  paths <- file.path(dir, paste0(names, suffix, ".png"))

  result <- shim()$write_masks(coloured, as.list(paths))
  if (!isTRUE(result$ok)) abort_engine(result)
  invisible(unlist(result$paths))
}
