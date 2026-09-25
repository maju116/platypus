# Training, evaluating and predicting.
#
# One call trains every model in the specification, because comparing two architectures on
# the same data is the question people actually have, and doing it by hand means writing
# the same loop twice and hoping the splits matched.

#' Train the models in a specification
#'
#' Trains every model the specification asks for, in order, each with its own data
#' pipeline - so two models may differ in input size or tiling and still be compared
#' fairly, on the same data.
#'
#' Validation data is never augmented. Measuring a model on distorted images measures the
#' distortion.
#'
#' This blocks until training finishes. With `verbose = TRUE` each epoch is reported as it
#' completes, which on anything longer than a few minutes is the difference between
#' waiting and wondering.
#'
#' @param spec A [platypus_spec()].
#' @param device `"cuda"`, `"cpu"`, or `NULL` to use a GPU when one is present.
#' @param num_workers Processes used to read and decode images. `"auto"` picks a small
#'   number based on the machine. This matters more than it sounds: reading in the main
#'   process alone leaves the GPU waiting on the disk, and on the Data Science Bowl at
#'   128x128 that is the difference between 35 seconds an epoch and under 10. Set `0` to
#'   avoid multiprocessing entirely if it causes trouble.
#' @param strict_data Treat an incomplete sample as an error. Set `FALSE` to train on the
#'   rest and be told what was skipped.
#' @param verbose Report each epoch as it finishes.
#' @return A `platypus_fit`.
#' @export
#' @examples
#' \dontrun{
#' fit <- platypus_fit(spec, verbose = TRUE)
#' evaluate(fit)
#' masks <- predict(fit, "unet")
#' }
platypus_fit <- function(spec, device = NULL, num_workers = "auto", strict_data = TRUE,
                         verbose = FALSE) {
  if (!inherits(spec, "platypus_spec")) {
    stop("`spec` must come from `platypus_spec()`; see `?platypus_spec`.", call. = FALSE)
  }

  built <- shim()$build_engine(spec$py, device = device,
                               num_workers = resolve_workers(num_workers),
                               strict_data = strict_data)
  if (!isTRUE(built$ok)) abort_engine(built)

  # Said before the wait, not after: a run that quietly fell back to the processor looks
  # exactly like a run that is simply slow, and the difference is an hour.
  device <- shim()$device_report()
  if (isTRUE(device$gpu_present_but_unusable)) {
    warning(
      "Training on the processor: this machine has a GPU, but the installed PyTorch ",
      "build\n  cannot use it. For a GTX 10-series card or older, restart R and call ",
      "platypus_use_torch(\"pascal\").\n  See ?platypus_device.",
      call. = FALSE, immediate. = TRUE
    )
  }

  result <- shim()$run_fit(built$engine, verbose = verbose)
  if (!isTRUE(result$ok)) abort_engine(result)

  structure(
    list(
      engine = built$engine,
      spec = spec,
      models = unlist(result$models),
      history = rows_to_frame(result$history),
      stop_reasons = result$stop_reasons,
      device = device
    ),
    class = "platypus_fit"
  )
}

#' @export
print.platypus_fit <- function(x, ...) {
  cat("platypus fit\n")
  cat("  models   :", paste(x$models, collapse = ", "), "\n")
  cat("  device   :", x$device$device, paste0("(torch ", x$device$torch, ")"), "\n")
  if (nrow(x$history)) {
    epochs <- tapply(x$history$epoch, x$history$model, max)
    cat("  epochs   :", paste(sprintf("%s=%d", names(epochs), epochs), collapse = ", "), "\n")
  }
  for (model in names(x$stop_reasons)) {
    cat("  stopped  :", model, "-", x$stop_reasons[[model]], "\n")
  }
  cat("  next     : evaluate(fit), predict(fit, \"", x$models[[1]], "\")\n", sep = "")
  invisible(x)
}

#' Compare the trained models
#'
#' One row per model: what it is, how big it is, how long it trained, and how it scored.
#'
#' The `loss` column is only comparable between models trained on the same loss, which is
#' why `loss_function` is there beside it. A Focal-Tversky of 0.05 is not better than a
#' CCE-Dice of 0.14; it is not even the same question. Metrics stay comparable, because
#' they measure the mask rather than the objective.
#'
#' @param object A [platypus_fit()].
#' @param split Which data to score on: `"validation"`, `"train"` or `"test"`.
#' @param ... Unused.
#' @return A data frame, one row per model.
#' @export
evaluate <- function(object, ...) UseMethod("evaluate")

#' @rdname evaluate
#' @export
evaluate.platypus_fit <- function(object, split = "validation", ...) {
  result <- shim()$evaluation_table(object$engine, split = split)
  if (!isTRUE(result$ok)) abort_engine(result)
  rows_to_frame(result$table)
}

#' Predict masks
#'
#' @param object A [platypus_fit()].
#' @param model Which model, by the name given in the specification. Defaults to the
#'   first one trained.
#' @param split Which data to predict on.
#' @param type `"class"` gives the class index per pixel, which is the mask to look at;
#'   `"probability"` keeps the per-class channel.
#' @param ... Unused.
#' @return For `"class"`, an integer array of `image x height x width`, with classes
#'   numbered from 1 to match the colormap. For `"probability"`, the same with a trailing
#'   class axis. Tiled models return images at their original size, reassembled.
#' @export
predict.platypus_fit <- function(object, model = NULL, split = "test",
                                 type = c("class", "probability"), ...) {
  type <- match.arg(type)
  model <- model %||% object$models[[1]]
  if (!model %in% object$models) {
    stop("no model called '", model, "'; this fit has: ",
         paste(object$models, collapse = ", "), call. = FALSE)
  }
  result <- shim()$predictions(object$engine, model, split = split,
                               as_class = identical(type, "class"))
  if (!isTRUE(result$ok)) abort_engine(result)
  result$masks
}

#' The epoch-by-epoch record
#'
#' @param object A [platypus_fit()].
#' @return A data frame with one row per model per epoch.
#' @export
training_history <- function(object) {
  if (!inherits(object, "platypus_fit")) {
    stop("`object` must come from `platypus_fit()`.", call. = FALSE)
  }
  object$history
}

#' How many processes should read the data
#'
#' Zero was the obvious default and the wrong one. Decoding a few hundred PNGs in the main
#' process leaves the GPU idle most of the epoch: measured on the Data Science Bowl at
#' 128x128, zero workers took 35 seconds an epoch and eight took under ten. A package that
#' promises a quick first result should not ship the slow arrangement by default.
#'
#' Four is a deliberate ceiling rather than a count of the cores. The work is reading and
#' decoding files, so it saturates well before the processor does, and each worker holds
#' its own copy of the sample cache.
#' @keywords internal
#' @noRd
resolve_workers <- function(x) {
  if (is.numeric(x)) return(as.integer(x))
  if (!identical(x, "auto")) {
    stop("`num_workers` must be a number or \"auto\".", call. = FALSE)
  }
  cores <- tryCatch(parallel::detectCores(logical = FALSE), error = function(e) NA_integer_)
  if (is.na(cores) || cores < 2L) return(0L)
  min(4L, cores - 1L)
}

#' Turn a list of equally-shaped records into a data frame
#'
#' The engine reports tables as lists of rows, which is what survives the crossing
#' intact. Missing entries become NA rather than shifting a column.
#' @keywords internal
#' @noRd
rows_to_frame <- function(rows) {
  if (!length(rows)) return(data.frame())
  columns <- unique(unlist(lapply(rows, names)))
  frame <- lapply(columns, function(column) {
    values <- lapply(rows, function(row) {
      value <- row[[column]]
      if (is.null(value) || !length(value)) NA else value[[1]]
    })
    unlist(values)
  })
  names(frame) <- columns
  as.data.frame(frame, stringsAsFactors = FALSE)
}
