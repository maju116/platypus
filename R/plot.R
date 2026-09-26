# Pictures.
#
# The package's reason for existing, as much as the training is. Getting from a folder of
# images to a figure that can go in a paper is the job; a model is the middle of it.

#' Read images as the model sees them
#'
#' Reads through the engine's own pipeline rather than a separate decoder, so what you
#' plot is what the model was given. A picture drawn from a differently resized copy is a
#' picture of something the model never saw, and the disagreements it appears to show may
#' be nothing but the resizing.
#'
#' @param paths Image files.
#' @param size `c(height, width)` to read at, or `NULL` to keep each image's own size -
#'   which only works if they all agree.
#' @param channels 3 for colour, 1 for greyscale.
#' @param nearest Use nearest-neighbour resizing. Required for masks: interpolating one
#'   invents colours belonging to no class, which then quietly become background.
#' @return An `image x height x width x channels` array of 0-255 values.
#' @export
#' @examples
#' \dontrun{
#' images <- read_images(list.files("test/", full.names = TRUE), size = c(256, 256))
#' }
read_images <- function(paths, size = NULL, channels = 3, nearest = FALSE) {
  result <- shim()$read_images(as.list(as.character(paths)),
                               channels = as.integer(channels),
                               size = if (is.null(size)) NULL else as.integer(size),
                               nearest = nearest)
  if (!isTRUE(result$ok)) abort_engine(result)
  result$images
}

#' Look at masks beside the images they came from
#'
#' Builds a grid: one row per image, one column per panel. Which panels appear depends on
#' what you pass - the image alone, the truth, the prediction, and where they disagree.
#'
#' The disagreement panel is usually the one worth looking at. It separates what was found
#' from what was missed and what was invented, and those are not interchangeable: for most
#' clinical questions a missed lesion and a false alarm cost different things, and a single
#' overlap score hides which one you have.
#'
#' @param images An `image x height x width x channels` array, or one
#'   `height x width x channels` image.
#' @param prediction Predicted class indices, from [predict.platypus_fit()].
#' @param truth True class indices, if you have them.
#' @param colormap A list of RGB triples, one per class.
#' @param which Which images to show.
#' @param alpha How strongly to tint the overlays.
#' @param labels Row labels; defaults to the image number.
#' @param slice For volumes, which slice to draw. A volume cannot honestly be shown as one
#'   picture, so one plane is chosen and said out loud rather than a projection being
#'   invented. Counted along the last axis, which after the canonical reorientation is the
#'   axial direction - the view a radiologist scrolls through. `"middle"` takes the middle
#'   slice, which is the sensible first look.
#' @return A `ggplot`.
#' @export
#' @examples
#' \dontrun{
#' plot_masks(images, prediction = masks, truth = truth, colormap = binary_colormap)
#' }
plot_masks <- function(images, prediction = NULL, truth = NULL,
                       colormap = binary_colormap, which = NULL,
                       alpha = 0.55, labels = NULL, slice = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("plot_masks() needs the ggplot2 package.", call. = FALSE)
  }

  if (is_volumetric(images)) {
    if (is.null(slice)) {
      stop("these look like volumes (", paste(dim(images), collapse = " x "), "). ",
           "Choose a slice to draw - `slice = \"middle\"` to start - because a volume ",
           "shown as one picture is either a lie or a projection nobody asked for.",
           call. = FALSE)
    }
    images <- take_slice(images, slice, channels = TRUE)
    prediction <- take_slice(prediction, slice, channels = FALSE)
    truth <- take_slice(truth, slice, channels = FALSE)
  } else if (!is.null(slice)) {
    stop("`slice` only applies to volumes; these are ", length(dim(images)) - 2L,
         "D images.", call. = FALSE)
  }

  images <- as_image_stack(images)
  if (is.null(which)) which <- seq_len(min(4L, n_images(images)))
  which <- as.integer(which)

  panels <- "image"
  if (!is.null(truth)) panels <- c(panels, "truth")
  if (!is.null(prediction)) panels <- c(panels, "prediction")
  if (!is.null(truth) && !is.null(prediction)) panels <- c(panels, "agreement")

  rows <- lapply(which, function(i) {
    image <- as_rgb(images[i, , , , drop = FALSE][1, , , , drop = TRUE])
    lapply(panels, function(panel) {
      switch(panel,
        image = as_rgb(image),
        truth = mask_colours(slice_mask(truth, i), colormap),
        prediction = mask_colours(slice_mask(prediction, i), colormap),
        agreement = overlay_agreement(image, slice_mask(prediction, i),
                                      slice_mask(truth, i), alpha = alpha)
      )
    })
  })

  montage <- bind_panels(rows)
  height <- dim(montage)[1]
  width <- dim(montage)[2]
  tile_h <- height / length(which)
  tile_w <- width / length(panels)

  ggplot2::ggplot() +
    ggplot2::annotation_raster(
      grDevices::as.raster(montage / 255),
      xmin = 0, xmax = width, ymin = 0, ymax = height, interpolate = FALSE
    ) +
    ggplot2::scale_x_continuous(
      limits = c(0, width), expand = c(0, 0),
      breaks = (seq_along(panels) - 0.5) * tile_w, labels = panels, position = "top"
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, height), expand = c(0, 0),
      breaks = height - (seq_along(which) - 0.5) * tile_h,
      labels = labels %||% paste("image", which)
    ) +
    ggplot2::coord_fixed() +
    ggplot2::labs(x = NULL, y = NULL) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank()
    )
}

#' Plot what happened during training
#'
#' One panel per quantity, one line per model. Useful for the thing a final number cannot
#' tell you: whether it had stopped improving, or was still going when it ran out of
#' epochs.
#'
#' @param x A [platypus_fit()].
#' @param metrics Which columns to plot; defaults to the losses and metrics.
#' @param ... Unused.
#' @return A `ggplot`.
#' @importFrom rlang .data
#' @export
plot.platypus_fit <- function(x, metrics = NULL, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("plotting a fit needs the ggplot2 package.", call. = FALSE)
  }
  history <- training_history(x)
  if (!nrow(history)) stop("this fit has no history to plot.", call. = FALSE)

  ignore <- c("model", "epoch", "seconds", "learning_rate")
  metrics <- metrics %||% setdiff(names(history), ignore)

  long <- do.call(rbind, lapply(metrics, function(metric) {
    data.frame(model = history$model, epoch = history$epoch,
               quantity = metric, value = history[[metric]],
               stringsAsFactors = FALSE)
  }))
  # train_loss and val_loss belong on the same panel; train_dice and val_dice likewise.
  long$panel <- sub("^(train|val)_", "", long$quantity)
  long$split <- ifelse(grepl("^train_", long$quantity), "train", "validation")

  ggplot2::ggplot(long, ggplot2::aes(x = .data$epoch, y = .data$value,
                                     colour = .data$model,
                                     linetype = .data$split)) +
    ggplot2::geom_line(linewidth = 0.7) +
    ggplot2::geom_point(size = 1.2) +
    ggplot2::facet_wrap(~ panel, scales = "free_y") +
    # Epochs are counted, so half of one is not a thing. The default breaks offer 1.5 and
    # 2.5 on a short run, which is the sort of detail that makes a figure look unconsidered.
    ggplot2::scale_x_continuous(breaks = function(limits) {
      unique(as.integer(round(pretty(limits))))
    }) +
    ggplot2::labs(x = "epoch", y = NULL, colour = NULL, linetype = NULL) +
    ggplot2::theme_minimal(base_size = 11)
}

#' @keywords internal
#' @noRd
n_images <- function(images) {
  if (length(dim(images)) == 4L) dim(images)[1] else 1L
}

#' @keywords internal
#' @noRd
as_image_stack <- function(images) {
  if (length(dim(images)) == 3L) array(images, dim = c(1L, dim(images))) else images
}

#' @keywords internal
#' @noRd
slice_mask <- function(mask, i) {
  if (length(dim(mask)) == 3L) mask[i, , ] else mask
}

#' Lay panels out as one image: rows of samples, columns of views
#' @keywords internal
#' @noRd
bind_panels <- function(rows) {
  row_arrays <- lapply(rows, function(panels) {
    Reduce(function(a, b) abind_axis(a, b, 2L), panels)
  })
  Reduce(function(a, b) abind_axis(a, b, 1L), row_arrays)
}

#' Join two arrays along one axis, without taking a dependency to do it
#' @keywords internal
#' @noRd
abind_axis <- function(a, b, axis) {
  stopifnot(length(dim(a)) == 3L, length(dim(b)) == 3L)
  if (axis == 1L) {
    out <- array(0L, dim = c(dim(a)[1] + dim(b)[1], dim(a)[2], 3L))
    out[seq_len(dim(a)[1]), , ] <- a
    out[dim(a)[1] + seq_len(dim(b)[1]), , ] <- b
  } else {
    out <- array(0L, dim = c(dim(a)[1], dim(a)[2] + dim(b)[2], 3L))
    out[, seq_len(dim(a)[2]), ] <- a
    out[, dim(a)[2] + seq_len(dim(b)[2]), ] <- b
  }
  out
}


#' Does this array hold volumes rather than images?
#'
#' Read off the rank: a stack of images is (n, h, w, c) and a stack of volumes is
#' (n, d, h, w, c). One volume, (d, h, w, c), has the same rank as a stack of images, and
#' that ambiguity is real - it is resolved by the caller passing `slice`, which is why an
#' unsliced volume is an error rather than a guess.
#' @keywords internal
#' @noRd
is_volumetric <- function(images) {
  length(dim(images)) == 5L
}

#' Take one plane out of a volume, along the last spatial axis
#' @keywords internal
#' @noRd
take_slice <- function(x, slice, channels) {
  if (is.null(x)) return(NULL)
  dims <- dim(x)
  rank <- length(dims)
  # Which axis is the slice: the last spatial one, which is the last axis for masks and the
  # one before the channels for images.
  axis <- if (channels) rank - 1L else rank
  depth <- dims[axis]

  index <- if (identical(slice, "middle")) {
    as.integer(ceiling(depth / 2))
  } else {
    as.integer(slice)
  }
  if (is.na(index) || index < 1L || index > depth) {
    stop("slice ", slice, " is outside 1..", depth, ".", call. = FALSE)
  }

  selector <- rep(list(quote(expr = )), rank)
  selector[[axis]] <- index
  out <- do.call(`[`, c(list(x), selector, list(drop = FALSE)))
  dim(out) <- dims[-axis]
  out
}
