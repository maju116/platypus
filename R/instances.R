#' Data for instance segmentation: tell the objects apart, not just find them
#'
#' Masks stored one file per object, read as a label map that keeps each object's identity
#' instead of uniting them into one foreground. The question this answers is *how many
#' nuclei, and how big is each one*, which a semantic mask cannot answer however good it is:
#' two touching nuclei are one region in it.
#'
#' **Why this exists rather than an argument on [segmentation_data()].** Every field
#' `instance_data()` takes, that one takes too - what makes it a different constructor is
#' the five it **refuses**: `colormap`, `labels`, `regions`, `channels_from` and
#' `target_spacing` do not apply to a directory of per-object masks, and the engine rejects
#' them by name. A constructor whose contract is "these five do not apply" is a contract
#' rather than a label, and it gets that refusal from its own argument list rather than from
#' a table somebody has to keep in step.
#'
#' The consequence worth knowing before you choose: `task` is still read off the
#' constructors rather than typed. Calling this one *is* how a specification says it is
#' about instances, so there is no `task = "instance_segmentation"` to write and nothing to
#' keep in agreement with it. See [platypus_spec()].
#'
#' What the model does with this is the other half, and it is one block: a model's
#' `separation` says what the network predicts - `rim` or `flows` - and `decode` says how
#' those channels become objects. Neither belongs here, because they describe the run and
#' not the data.
#'
#' @param train Training data: a directory, a configuration file, or a `platypus_split`
#'   from [split_dataset()], which carries its own validation set and division.
#' @param validation A validation set, or `FALSE` for none. Exactly one of `validation` and
#'   `split`, and neither when `train` is already a `platypus_split`.
#' @param test A test set to predict on. Scored only if it carries masks.
#' @param split Divide `train` instead of naming a `validation` set: a list with
#'   `fractions` (two numbers, or three to cut a test set too) and `group_by`, which is
#'   required and may be `NULL`. Exactly one of `split` and `validation`.
#' @param mode `"nested_dirs"` or `"config_file"`.
#' @param window How values in real units reach the model, for DICOM and NIfTI. See
#'   [segmentation_data()].
#' @param subdirs For `nested_dirs`, the names of the image and mask subdirectories. The
#'   second one holds **one file per object** rather than one mask per image, which is the
#'   layout this task is named for.
#' @param column_sep For `config_file`, what separates several paths in one cell.
#' @param shuffle Shuffle the training data between epochs.
#' @return A data specification, to be passed to [platypus_spec()].
#' @seealso [segmentation_data()] for one mask per image, [platypus_spec()]
#' @export
#' @examples
#' instance_data("stage1_train/", "stage1_validation/")
instance_data <- function(train, validation = NULL, test = NULL, split = NULL,
                          mode = c("nested_dirs", "config_file"),
                          window = NULL,
                          subdirs = c("images", "masks"), column_sep = ";",
                          shuffle = TRUE) {
  # Which of these the caller actually named: a field is sent only when somebody chose it,
  # because a default sent from here is indistinguishable from a value the user set and an
  # R-side default silences a schema that knows better. §3D's rule, and the same `gave`
  # list `segmentation_data()` keeps for the same reason.
  explicit_mode <- !missing(mode)
  gave <- list(
    subdirs = !missing(subdirs),
    column_sep = !missing(column_sep),
    shuffle = !missing(shuffle)
  )
  mode <- match.arg(mode)

  # A split goes straight in, exactly as it does for semantic data: making the caller
  # unpack three paths and remember to switch to config_file mode would be three chances to
  # get it wrong, in the one place where a mistake means training and validating on the
  # same patient.
  if (inherits(train, "platypus_split")) {
    if (!is.null(validation)) {
      stop("`train` is already a platypus_split, which carries the validation set; ",
           "leave `validation` unset.", call. = FALSE)
    }
    if (!is.null(split)) {
      stop("`train` is already a platypus_split, so it carries the division; `split` ",
           "would divide it a second time. Give one or the other.", call. = FALSE)
    }
    done <- train
    train <- done$train_path
    validation <- done$validation_path
    if (is.null(test)) test <- done$test_path
    if (!explicit_mode) mode <- "config_file"
  }

  check_split(validation, split, test)

  structure(
    list(
      train_path = train,
      # FALSE is not a path. It travels as the engine's own `validation` flag and
      # `validation_path` stays absent, because sending both is the contradiction the
      # engine refuses.
      validation_path = if (isFALSE(validation)) NULL else validation,
      validation = if (isFALSE(validation)) FALSE else NULL,
      test_path = test,
      # Not compacted: `group_by = NULL` is the whole point of that field - "divide by
      # file, and I know it" - so an absent value and a NULL value are different things.
      split = split_block(split),
      mode = if (explicit_mode || !identical(mode, "nested_dirs")) mode else NULL,
      # Aligned under the `if` it continues: the chain reads as one decision.
      window = if (is.null(window)) NULL
               else if (is.character(window)) window  # nolint: indentation_linter.
               else as.numeric(window),
      subdirs = if (gave$subdirs) as.character(subdirs) else NULL,
      column_sep = if (gave$column_sep) column_sep else NULL,
      shuffle = if (gave$shuffle) shuffle else NULL
    ),
    class = c("platypus_instance_data", "list"),
    # The tag is the whole mechanism: `task_of()` reads it, `agreed_task()` checks it
    # against the models, and nothing else in a specification has to say the word.
    task = "instance_segmentation"
  )
}

#' What the network predicts, so objects can be told apart
#'
#' One per model, and it decides the head's width and which losses are admissible - both
#' derived by the engine rather than stated here, so there is no channel count to get wrong.
#' Pass the result as a model's `separation`; see [u_net()].
#'
#' `separation_rim()` trains three classes - background, object interior, object rim - and
#' the rim is what separates two touching objects. `separation_flows()` trains a vector
#' field pointing at each object's own sink, so every foreground pixel carries a value and
#' none of them is spent on being a boundary.
#'
#' **What this is worth was measured and it is not flattering.** On the Data Science Bowl's
#' held-out cases the flow route is worth +0.0680 of instance average precision over
#' `decode_connected_components()`, which ignores the signal entirely, and the rim route
#' +0.0640 - while substituting a *perfect mask* is worth +0.2398. So the mask is the limit
#' by a factor of 3.6, and a run that beats the published weights' 0.4888 without reaching
#' 0.7655 has improved its mask and said nothing about separation.
#'
#' There is no `hv`. Its markers lose to a one-pixel erosion on perfect inputs, 0.9382
#' against 0.9807, so naming it would promise a mechanism that was measured as absent.
#'
#' @param width For `rim`, how many pixels wide the rim is, **in the model's input
#'   resolution**. Unset uses the engine's default, which is 2 and stands on an argument
#'   rather than on a measurement: the published sweep peaked at 3 pixels of *source*
#'   resolution, which on a 520-pixel case read at 256 is about 1.5 pixels of model input,
#'   and a sweep in model pixels has not been run.
#' @param mode For `flows`, `"diffusion"` or `"centroid"`. Unset uses the engine's default
#'   of `"diffusion"`, which stands on a measurement: decoded against the truth it reaches
#'   0.9924 where `"centroid"` reaches 0.9478, because a straight line to a centroid leaves
#'   a crescent-shaped object.
#' @param loss For `flows`, the objective for the vector channels, which has to be one
#'   defined over numbers rather than over classes. Unset uses the engine's default.
#' @param weight For `flows`, how much of the total objective the vector half provides.
#' @return A separation block, to be passed as a model's `separation`.
#' @seealso [decodings], [instance_data()]
#' @name separations
#' @examples
#' separation_rim(width = 2)
#' separation_flows(mode = "diffusion")
NULL

#' @rdname separations
#' @export
separation_rim <- function(width = NULL) {
  compact(list(name = "rim", width = int1(width)))
}

#' @rdname separations
#' @export
separation_flows <- function(mode = NULL, loss = NULL, weight = NULL) {
  compact(list(name = "flows", mode = mode, loss = loss, weight = weight))
}

#' How predicted channels become objects
#'
#' A model's `decode` is a **list** of these, not one: several decodings share one forward
#' pass, so a second one costs seconds where a second training run costs an hour.
#' [evaluate()] then returns a row per model and decoding.
#'
#' Each declares which separation it can read, and asking for one that cannot read the
#' model's signal is refused when the specification is built rather than at the end of a
#' run. `decode_connected_components()` reads none of them - it ignores the separation
#' signal entirely - which is exactly why it belongs in every model's list: it is the row
#' the signal's own row has to be read against. Left unset, a model decodes with its own
#' signal's decoding plus that yardstick.
#'
#' @param minimum_area Drop an object smaller than this many pixels. Unset keeps
#'   everything, and that is the engine's default on purpose: the right floor is a property
#'   of the data rather than of the method. A nucleus at 256x256 in the Data Science Bowl
#'   runs from 4 pixels up, so a floor picked here would silently delete the smallest real
#'   objects in one dataset while letting speckle through in another. It is applied after
#'   the objects are formed, so it removes answers rather than changing how they were
#'   reached.
#' @param steps For `decode_follow_flows()`, how many Euler steps each pixel takes.
#'   **Measured as not mattering**: one step scores 0.9914 against 0.9923 converged, a span
#'   below the 0.0013 two runs differ by, because the field is continuous within an object
#'   and diverges at a seam - so the gap between two objects opens on the first step rather
#'   than after many. Adjustable because that argument is about this data's object sizes.
#' @param step_size For `decode_follow_flows()`, how far one step moves, in pixels, on a
#'   unit-length field. **A cliff rather than a speed knob**: 0.9923 at 1, 0.4715 at 2,
#'   0.0303 at 10. A step longer than a pixel jumps over the sink it was converging on and
#'   lands in a neighbour's basin.
#' @return A decoding block, to be passed inside a model's `decode` list.
#' @seealso [separations], [instance_data()]
#' @name decodings
#' @examples
#' list(decode_nearest_seed(), decode_connected_components())
#' decode_follow_flows(step_size = 0.5)
NULL

#' @rdname decodings
#' @export
decode_connected_components <- function(minimum_area = NULL) {
  compact(list(name = "connected_components", minimum_area = int1(minimum_area)))
}

#' @rdname decodings
#' @export
decode_nearest_seed <- function(minimum_area = NULL) {
  compact(list(name = "nearest_seed", minimum_area = int1(minimum_area)))
}

#' @rdname decodings
#' @export
decode_watershed <- function(minimum_area = NULL) {
  compact(list(name = "watershed", minimum_area = int1(minimum_area)))
}

#' @rdname decodings
#' @export
decode_follow_flows <- function(steps = NULL, step_size = NULL, minimum_area = NULL) {
  compact(list(
    name = "follow_flows",
    steps = int1(steps),
    # Not an integer: this is a distance in pixels and the measurement that matters is
    # below 1. `int1()` here would turn `0.5` into `0`, which is a decoding that never
    # moves rather than an error anybody could see.
    step_size = step_size,
    minimum_area = int1(minimum_area)
  ))
}
