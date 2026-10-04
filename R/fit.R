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
#' @param check_masks Read a sample of the training masks before training, and refuse when
#'   a class the specification declares appears in none of them. A colormap or set of
#'   labels that matches none of the labelled tissue is the quietest way a run wastes a
#'   day: every mask reads as background, the loss falls because background is most of a
#'   medical image, and the model learns to answer "nothing here". Set `FALSE` only when
#'   the missing class is real but rarer than the sample - and see [mask_report()] first,
#'   which asks the same question over as much of the data as you like.
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
                         check_masks = TRUE, verbose = FALSE) {
  if (!inherits(spec, "platypus_spec")) {
    stop("`spec` must come from `platypus_spec()`; see `?platypus_spec`.", call. = FALSE)
  }

  built <- shim()$build_engine(spec$py, device = device,
                               num_workers = resolve_workers(num_workers),
                               strict_data = strict_data,
                               check_masks = isTRUE(check_masks))
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
      task = spec_task(spec),
      models = unlist(result$models),
      history = rows_to_frame(result$history),
      stop_reasons = result$stop_reasons,
      device = device
    ),
    class = "platypus_fit"
  )
}

#' Which task a built specification describes
#'
#' Read from the Python object rather than remembered from the R one, because a
#' specification can also arrive from a YAML file where no R constructor was involved.
#'
#' @noRd
spec_task <- function(spec) {
  task <- tryCatch(shim()$spec_as_dict(spec$py)$task, error = function(e) NULL)
  if (is.null(task) || !nzchar(task)) "segmentation" else as.character(task)
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
  result <- if (identical(object$task, "detection")) {
    shim()$detection_table(object$engine, split = split)
  } else {
    shim()$evaluation_table(object$engine, split = split)
  }
  if (!isTRUE(result$ok)) abort_engine(result)
  rows_to_frame(result$table)
}

#' Predict masks
#'
#' @section Which grid the answer is on:
#'
#' `space = "model"` returns everything on the grid the model worked at, stacked into one
#' array. That is what training saw, and it is the only form that can be a single array,
#' because a stack needs one shape.
#'
#' `space = "source"` maps each prediction back onto the grid of the scan it was computed
#' from - undoing the resampling and the crop - and therefore returns a **list**, one mask per
#' scan. This is what you want for anything that has to meet the original data: laying a mask
#' over the scan in a viewer, writing it beside the scan with [save_volumes()], or measuring it
#' in millilitres of the scan's own voxels. Scans differ in size, so the list is not a
#' limitation being worked around - resizing them to a common shape is exactly how a mask ends
#' up describing anatomy it was not computed from.
#'
#' Two things are lost mapping back, and neither can be recovered. Interpolation softens a
#' boundary, so the mask returns slightly smoother than the model drew it. And where reading
#' cropped anatomy away, the mask comes back padded with background - which means *not
#' examined*, not *nothing there*. Give the model an `input_shape` that covers the anatomy if
#' that distinction matters.
#'
#' @param object A [platypus_fit()].
#' @param model Which model, by the name given in the specification. Defaults to the
#'   first one trained.
#' @param split Which data to predict on.
#' @param type `"class"` gives the class index per pixel, which is the mask to look at;
#'   `"probability"` keeps the per-class channel.
#' @param space `"model"` or `"source"`; see above. Note that the two return different
#'   shapes of thing, an array and a list, because they are different things.
#' @param ... Unused.
#' @return With `space = "model"`: for `"class"`, an integer array of
#'   `image x height x width`, classes numbered from 1 to match the colormap; for
#'   `"probability"`, the same with a trailing class axis. Tiled models return images at their
#'   original size, reassembled.
#'
#'   With `space = "source"`: a list of such arrays, one per scan, each on that scan's grid.
#' @export
predict.platypus_fit <- function(object, model = NULL, split = "test",
                                 type = c("class", "probability"),
                                 space = c("model", "source"), ...) {
  # Asked before `match.arg` assigns to them: in R, `missing()` on a formal argument stops
  # being true once the argument has been assigned to, so reading it after the two lines
  # below reports "the caller passed this" for every caller. The first version did exactly
  # that and refused the plain `predict(fit)` that every user makes first.
  asked_type <- !missing(type)
  asked_space <- !missing(space)
  type <- match.arg(type)
  space <- match.arg(space)
  model <- model %||% object$models[[1]]
  if (!model %in% object$models) {
    stop("no model called '", model, "'; this fit has: ",
         paste(object$models, collapse = ", "), call. = FALSE)
  }
  if (identical(object$task, "detection")) {
    if (asked_type || asked_space) {
      stop("`type` and `space` are about masks. A detector returns boxes, always in each ",
           "image's own pixels - there is no other space they could be in that anyone ",
           "would want.", call. = FALSE)
    }
    found <- shim()$detections(object$engine, model, split = split)
    if (!isTRUE(found$ok)) abort_engine(found)
    return(as_box_frames(found$detections))
  }

  result <- shim()$predictions(object$engine, model, split = split,
                              as_class = identical(type, "class"), space = space)
  if (!isTRUE(result$ok)) abort_engine(result)
  result$masks
}

#' One data frame of boxes per image, named by the sample key
#'
#' A data frame rather than a matrix, because a box carries a score and a class name as
#' well as four numbers, and because that is the shape `plot_boxes()` and `write.csv()`
#' both want. An image with nothing found gets a frame with no rows rather than being
#' dropped: a split of fifty images returns fifty entries, so nothing downstream has to
#' line names up against a shorter list.
#'
#' @noRd
as_box_frames <- function(found) {
  frames <- lapply(found, function(entry) {
    boxes <- entry$boxes
    if (is.null(boxes) || !length(boxes)) {
      return(data.frame(xmin = numeric(0), ymin = numeric(0), xmax = numeric(0),
                        ymax = numeric(0), score = numeric(0), label = integer(0),
                        name = character(0), stringsAsFactors = FALSE))
    }
    boxes <- matrix(as.numeric(boxes), ncol = 4)
    data.frame(
      xmin = boxes[, 1], ymin = boxes[, 2], xmax = boxes[, 3], ymax = boxes[, 4],
      score = as.numeric(entry$scores),
      label = as.integer(entry$labels),
      name = as.character(entry$names),
      stringsAsFactors = FALSE
    )
  })
  names(frames) <- vapply(found, function(entry) as.character(entry$key), character(1))
  frames
}

#' Average precision per class
#'
#' The row that matters on unbalanced data, which is most detection data: BCCD has 4,155
#' red cells against 372 white and 361 platelets, so a single number is a number about red
#' cells. [evaluate()] deliberately carries no overall precision or recall for the same
#' reason - averaging them over classes needs a weighting, and every choice of weighting is
#' a different claim.
#'
#' @param object A fit from [platypus_fit()] on a detection specification.
#' @param model Which model, when the specification trained several.
#' @param split Which split to score.
#' @param ... Unused.
#' @return A data frame, one row per class: average precision at IoU 0.5, the mean overlap
#'   of the boxes that matched, the number of true boxes, and precision and recall at the
#'   model's `operating_point`.
#' @export
evaluate_classes <- function(object, ...) UseMethod("evaluate_classes")

#' @rdname evaluate_classes
#' @export
evaluate_classes.platypus_fit <- function(object, model = NULL,
                                          split = "validation", ...) {
  if (!identical(object$task, "detection")) {
    stop("`evaluate_classes()` reports average precision per class, which is a detection ",
         "measure. For segmentation, `evaluate_cases()` reports per case and ",
         "`summarise_cases()` summarises it.", call. = FALSE)
  }
  model <- model %||% object$models[[1]]
  result <- shim()$detection_classes(object$engine, model, split = split)
  if (!isTRUE(result$ok)) abort_engine(result)
  rows_to_frame(result$rows)
}

#' The anchors a detector used, and whether they were fitted
#'
#' A detector cannot be reloaded without its anchors, and when they were fitted rather than
#' named in the specification this is the only record. [save_weights()] writes them into the
#' sidecar for the same reason.
#'
#' @param object A fit from [platypus_fit()] on a detection specification.
#' @param model Which model, when the specification trained several.
#' @return A list: `anchors` as three groups of pairs, `fitted` saying whether they were
#'   fitted to your data, and when they were, the mean overlap they achieve and a data
#'   frame with one row per anchor.
#' @export
detection_anchors <- function(object, model = NULL) {
  if (!inherits(object, "platypus_fit") || !identical(object$task, "detection")) {
    stop("`detection_anchors()` needs a fit from a detection specification.",
         call. = FALSE)
  }
  model <- model %||% object$models[[1]]
  result <- shim()$run_anchors(object$engine, model)
  if (!isTRUE(result$ok)) abort_engine(result)
  list(
    anchors = result$anchors,
    fitted = isTRUE(result$fitted),
    mean_iou = result$mean_iou,
    boxes_used = result$boxes_used,
    per_anchor = rows_to_frame(result$rows)
  )
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

#' Score every case separately
#'
#' `evaluate()` gives one row per model: one number for a whole set. This gives one row per
#' case - each image, or each patient when `group_by` is set - because one number hides the
#' distribution, and the distribution is usually the finding. A model averaging Dice 0.86
#' that scores 0.1 on three patients has a failure mode, and the mean is where it hides.
#'
#' `summary()` on the result gives the distribution: mean, standard deviation, median,
#' range, and which case scored worst.
#'
#' Tiled models are scored on the whole image, not on tiles. Overlaps are accumulated
#' across a case's tiles and the metric applied once, which is that image's score exactly;
#' averaging the tiles' scores would be a different number, and unkind to any case whose
#' object happens to straddle a boundary.
#'
#' @param object A [platypus_fit()].
#' @param model Which model, by the name in the specification. Defaults to the first.
#' @param split Which data to score on.
#' @param group_by A pattern picking a group out of each case's name, as in
#'   [platypus_split()]. With it, the rows are patients rather than slices: a patient's
#'   slices are pooled into one score, the way a volume would be.
#' @param ... Unused.
#' @return A data frame with one row per case, of class `platypus_cases`.
#' @seealso [platypus_split()] for keeping a patient out of two sets in the first place.
#' @examples
#' \dontrun{
#' cases <- evaluate_cases(fit)
#' summary(cases)
#' head(cases[order(cases$dice), ])          # the ones worth looking at
#'
#' evaluate_cases(fit, group_by = "^(patient\\\\d+)_")
#' }
#' @export
evaluate_cases <- function(object, ...) UseMethod("evaluate_cases")

#' @rdname evaluate_cases
#' @export
evaluate_cases.platypus_fit <- function(object, model = NULL, split = "validation",
                                        group_by = NULL, ...) {
  model <- model %||% object$models[[1]]
  if (!model %in% object$models) {
    stop("no model called '", model, "'; this fit has: ",
         paste(object$models, collapse = ", "), call. = FALSE)
  }
  result <- shim()$case_table(object$engine, model, split = split, group_by = group_by)
  if (!isTRUE(result$ok)) abort_engine(result)

  frame <- rows_to_frame(result$table)
  structure(frame, class = c("platypus_cases", class(frame)), label = result$label)
}

#' @param object A `platypus_cases` from [evaluate_cases()].
#' @rdname evaluate_cases
#' @export
summary.platypus_cases <- function(object, ...) {
  rows <- lapply(seq_len(nrow(object)), function(i) as.list(object[i, , drop = FALSE]))
  result <- shim()$case_summary(rows)
  if (!isTRUE(result$ok)) abort_engine(result)
  rows_to_frame(result$table)
}

#' @export
print.platypus_cases <- function(x, ...) {
  label <- attr(x, "label") %||% "case"
  cat("platypus scores by ", label, " (", nrow(x), " rows)\n", sep = "")
  print(as.data.frame(utils::head(x, 10)), row.names = FALSE)
  if (nrow(x) > 10) cat("  ... ", nrow(x) - 10, " more. summary() gives the distribution.\n", sep = "")
  invisible(x)
}

#' The published weights, and what they are
#'
#' Weights are named rather than downloaded by hand, and a name is pinned to one commit of the
#' repository holding it - so `"dsbowl-unet"` means one set of numbers today and the same set next
#' year. This lists what is published, with the commit each name resolves to.
#'
#' Read the descriptions rather than only the names. A model's limitations decide whether it
#' answers your question: the nuclei weights are semantic, so touching nuclei come back as one
#' region and a count taken from them would be wrong.
#'
#' @return A data frame with `name`, `repo`, `filename`, `revision` and `description`.
#' @seealso [u_net()] and the other architectures, whose `weights` argument takes a name from
#'   here, a `hf://owner/repo/file.safetensors@commit` reference, or a path to a local file.
#' @examples
#' \dontrun{
#' available_weights()
#'
#' spec <- platypus_spec(
#'   data = segmentation_data("images/", "images/", colormap = binary_colormap),
#'   models = list(u_net("nuclei", input_shape = c(256, 256), blocks = 4, filters = 16,
#'                       weights = "dsbowl-unet", fit = FALSE))
#' )
#' masks <- predict(platypus_fit(spec), split = "validation")
#' }
#' @export
available_weights <- function() {
  result <- shim()$weights_listing()
  if (!isTRUE(result$ok)) abort_engine(result)

  rows <- lapply(result$weights, function(row) {
    data.frame(name = row$name, repo = row$repo, filename = row$filename,
               revision = row$revision, description = row$description,
               stringsAsFactors = FALSE)
  })
  if (!length(rows)) {
    return(data.frame(name = character(), repo = character(), filename = character(),
                      revision = character(), description = character(),
                      stringsAsFactors = FALSE))
  }
  do.call(rbind, rows)
}

#' Save a trained model's weights
#'
#' Writes safetensors plus a `.json` sidecar recording what the weights are for: architecture,
#' input shape, channels, classes, and whatever else you pass. Loading refuses a model they do not
#' belong to, which is what the sidecar is for - weights trained on a different colormap with the
#' same number of classes load cleanly and predict nonsense.
#'
#' Anything in `...` joins the sidecar. The data they were trained on and its licence belong there:
#' weights whose provenance lives in somebody's memory cannot be used by anybody else, including
#' you in a year.
#'
#' @param object A [platypus_fit()].
#' @param path Where to write, with or without the `.safetensors` extension.
#' @param model Which model, by the name in the specification. Defaults to the first.
#' @param ... Extra fields for the sidecar, for instance `data = "BBBC038v1"`,
#'   `licence = "CC0 1.0"`.
#' @return The path written, invisibly, with the sidecar's contents attached as the `recorded`
#'   attribute - so you can see what was put beside the weights without opening the file.
#' @examples
#' \dontrun{
#' save_weights(fit, "weights/my-run",
#'              data = "our 2026 cohort", licence = "internal use only")
#' }
#' @export
save_weights <- function(object, path, model = NULL, ...) {
  if (!inherits(object, "platypus_fit")) {
    stop("`object` must come from `platypus_fit()`.", call. = FALSE)
  }
  model <- model %||% object$models[[1]]
  if (!model %in% object$models) {
    stop("no model called '", model, "'; this fit has: ",
         paste(object$models, collapse = ", "), call. = FALSE)
  }

  extra <- list(...)
  result <- shim()$save_weights(object$engine, model, path.expand(path),
                                extra = if (length(extra)) extra else NULL)
  if (!isTRUE(result$ok)) abort_engine(result)
  invisible(structure(result$path, recorded = result$recorded, sidecar = result$sidecar))
}


#' What the colormap or labels match in your masks
#'
#' Reads a sample of one split's masks and reports what the specification's colours, or
#' list of labels, actually matched. Trains nothing and loads no model.
#'
#' [platypus_fit()] does a small version of this before every run and refuses when a
#' declared class appears in none of the masks it read. This is the same question asked
#' deliberately, over as much of the data as you want - which is what to reach for when
#' that refusal looks wrong.
#'
#' @section What the numbers mean:
#'
#' `missing_classes` is the one to act on. A class that appears in no mask cannot be
#' learned: its channel of the target is zero everywhere, so there is no gradient towards
#' it and the model is never shown the thing it is being asked to find. Either the colours
#' do not describe these masks, or `n_class` counts a class the data does not contain.
#'
#' `unmatched` is the fraction of mask pixels matching no entry, which fall back to the
#' background class. **Read it knowing that it scales with the size of the thing being
#' segmented**, so it is loud for a large structure and almost silent for a small lesion.
#' Measured on masks that are white, with a colormap asking for a colour that is not
#' there: a foreground covering 20% of the image gives 19.8%, and one covering 0.6% gives
#' 0.61% - which is less than the 0.72% that JPEG compression leaves around the edge of a
#' perfectly correct mask. That is why the refusal is built on class presence instead.
#'
#' @param spec A [platypus_spec()].
#' @param split `"train"`, `"validation"` or `"test"`.
#' @param limit How many masks to read. They are spread across the split rather than
#'   taken from its start, because datasets arrive sorted and a prefix would answer about
#'   the beginning.
#' @return A one-row data frame: `unmatched`, `samples_checked`, `total_samples`, and
#'   `present_classes` and `missing_classes` as comma-separated strings so the frame
#'   stays printable. The classes are also attached as integer vectors in the attributes
#'   `present` and `missing`.
#' @export
#' @examples
#' \dontrun{
#' mask_report(spec)
#' mask_report(spec, split = "validation", limit = 2000)
#' }
mask_report <- function(spec, split = c("train", "validation", "test"), limit = 500) {
  if (!inherits(spec, "platypus_spec")) {
    stop("`spec` must come from `platypus_spec()`; see `?platypus_spec`.", call. = FALSE)
  }
  split <- match.arg(split)
  result <- shim()$mask_report(spec$py, split = split, limit = as.integer(limit))
  if (!isTRUE(result$ok)) abort_engine(result)

  present <- as.integer(unlist(result$present_classes))
  missing <- as.integer(unlist(result$missing_classes))
  out <- data.frame(
    split = split,
    unmatched = as.numeric(result$unmatched),
    samples_checked = as.integer(result$samples_checked),
    total_samples = as.integer(result$total_samples),
    present_classes = paste(present, collapse = ", "),
    # An empty string rather than NA: nothing missing is a result, not an absence of one.
    missing_classes = paste(missing, collapse = ", "),
    stringsAsFactors = FALSE
  )
  attr(out, "present") <- present
  attr(out, "missing") <- missing
  out
}
