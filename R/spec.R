# The specification: the one thing R and Python agree on.
#
# Built from arguments or read from a YAML file, and nothing downstream can tell which.
# That is the whole contract - a colleague can hand you their configuration file, you can
# hand them your R script, and both describe the same experiment.

#' Where the images and masks are
#'
#' Two layouts. `nested_dirs` expects one directory per sample containing an `images` and
#' a `masks` subdirectory - the Data Science Bowl arrangement, where a sample may hold
#' dozens of separate mask files that are combined into one. `config_file` expects a CSV
#' with `images` and `masks` columns, several paths per cell if needed, resolved relative
#' to the file itself so a configuration travels with its data.
#'
#' Classes are named one of two ways, and exactly one of them. `colormap` is for masks
#' stored as pictures: a list of RGB triples where position is the class index. `labels` is
#' for masks stored as label maps - a vector of the voxel values, in class order - which is
#' how every volume format stores them, and how a single-channel PNG can be read as well.
#'
#' A sample that is missing its masks is an error, not a warning - a warning in a loop over
#' five hundred directories is a warning nobody reads.
#'
#' @param train,validation Paths to the training and validation data. A
#'   [platypus_split()] may be given as `train` on its own: it carries all three
#'   paths and selects `config_file` mode, so a split needs no unpacking.
#' @param test Optional test data. Only images are read from it.
#' @param colormap A list of RGB triples, one per class, background first. For masks stored
#'   as pictures. Give this or `labels`, not both.
#' @param labels The voxel value of each class, in class order, for masks stored as label
#'   maps - `c(0, 1)` for a binary segmentation, `c(0, 1, 2)` where 1 is liver and 2 is
#'   tumour. This is how NIfTI and every other volume format label anything. Give this or
#'   `colormap`, not both.
#' @param mode `"nested_dirs"` or `"config_file"`.
#' @param window How values in real units reach the model. A named window - `"lung"`,
#'   `"soft_tissue"`, `"bone"`, `"brain"`, `"abdomen"`, `"liver"`, `"mediastinum"`,
#'   `"subdural"`, `"stroke"` - or an explicit `c(centre, width)` in Hounsfield units, or
#'   `"auto"` for the window the file recorded, which only DICOM has, or `"full"` for the
#'   whole range present. Applies to DICOM and to NIfTI; ignored for ordinary pictures,
#'   which are already 0-255.
#'
#'   Worth choosing rather than leaving: a fixed window is what makes two scans comparable.
#'   Scaling each scan by its own darkest and brightest voxel lets a single metal implant or
#'   marker rescale everything else in it.
#'
#'   `NULL`, the default, sends nothing and lets the engine decide - which also keeps
#'   ordinary image work running against an engine too old to know the setting exists.
#' @param dicom_window The former name of `window`, still accepted. It was accurate while
#'   DICOM was the only format that needed it.
#' @param channels_from For datasets that keep one channel per file: one pattern per channel,
#'   in channel order. BraTS ships four MRI sequences per patient - T1, T1 after contrast, T2
#'   and FLAIR - because different tumour structures are visible in different sequences;
#'   satellite sets keep their bands apart the same way.
#'
#'   The order is stated rather than inferred, and that is the point. Sorted, the BraTS names
#'   come out flair, t1, t1ce, t2 - reproducible and anatomically meaningless. A model trained
#'   with FLAIR in channel one and used on data whose channel one is T1 returns a plausible
#'   answer with the right shape and the right range, and nothing further down can notice.
#'
#'   Patterns are Python regular expressions, like `group_by` in [platypus_split()], and R's
#'   own quoting applies: `c("_t1\\.nii", "_t1ce\\.nii", "_t2\\.nii", "_flair\\.nii")`.
#'   Each must match exactly one of a sample's files - `"_t1"` would match both `_t1.nii.gz`
#'   and `_t1ce.nii.gz`, which is an error rather than a race - and every file must be claimed
#'   by some pattern, because one left out is data the model never sees.
#' @param target_spacing For volumes: resample every scan to this many millimetres per voxel,
#'   then centre-crop or pad to the model's `input_shape`. Without it a volume is simply
#'   resized into that shape, which is only harmless when every scan covers the same amount of
#'   the patient - and clinical scans do not. Forty slices of 1 mm is 40 mm of patient and
#'   forty of 2.5 mm is 100 mm, so resizing alone leaves the same organ a different size in
#'   each, and nothing in the data says so.
#'
#'   `c(1, 1, 1)` is the usual starting point. Ignored for 2D.
#' @param subdirs For `nested_dirs`, the names of the image and mask subdirectories.
#' @param column_sep For `config_file`, what separates several paths in one cell.
#' @param shuffle Shuffle the training data between epochs.
#' @return A data specification, to be passed to [platypus_spec()].
#' @export
#' @examples
#' segmentation_data("train/", "valid/", colormap = binary_colormap)
segmentation_data <- function(train, validation, colormap = NULL, labels = NULL,
                              test = NULL,
                              mode = c("nested_dirs", "config_file"),
                              window = NULL, dicom_window = NULL, target_spacing = NULL,
                              channels_from = NULL,
                              subdirs = c("images", "masks"), column_sep = ";",
                              shuffle = TRUE) {
  explicit_mode <- !missing(mode)
  mode <- match.arg(mode)

  # A split goes straight in. Making the caller unpack three paths and remember to switch
  # to config_file mode would be three chances to get it wrong, in the one place where a
  # mistake means training and validating on the same patient.
  if (inherits(train, "platypus_split")) {
    if (!missing(validation)) {
      stop("`train` is already a platypus_split, which carries the validation set; ",
           "leave `validation` unset.", call. = FALSE)
    }
    split <- train
    train <- split$train_path
    validation <- split$validation_path
    if (is.null(test)) test <- split$test_path
    if (!explicit_mode) mode <- "config_file"
  }

  # Caught here rather than in the engine, because the message is the same and this way it
  # costs nothing: no interpreter starts to tell someone they gave both or neither.
  if (is.null(colormap) == is.null(labels)) {
    stop("give exactly one of `colormap` (masks stored as pictures) and `labels` ",
         "(masks stored as label maps, which is how volumes do it).", call. = FALSE)
  }
  if (!is.null(window) && !is.null(dicom_window)) {
    stop("`dicom_window` is the former name of `window`; give one of them, not both.",
         call. = FALSE)
  }
  if (!is.null(dicom_window)) window <- dicom_window
  if (!is.null(channels_from) && length(channels_from) < 2L) {
    stop("`channels_from` needs one pattern per channel, so at least two. With a single ",
         "channel there is nothing to order.", call. = FALSE)
  }
  if (!is.null(channels_from) && anyDuplicated(channels_from)) {
    stop("`channels_from` patterns must be distinct - two channels matching the same file ",
         "would make one measurement into two.", call. = FALSE)
  }
  if (!is.null(target_spacing) &&
      (length(target_spacing) != 3L || anyNA(target_spacing) || any(target_spacing <= 0))) {
    stop("`target_spacing` must be three positive numbers, in millimetres.", call. = FALSE)
  }

  list(
    train_path = train,
    validation_path = validation,
    test_path = test,
    mode = mode,
    colormap = if (is.null(colormap)) NULL else lapply(colormap, as.integer),
    labels = if (is.null(labels)) NULL else as.integer(labels),
    # Left out entirely when unset. An older engine rejects fields it does not know, so
    # sending a default nobody asked for would break every run that has nothing to do
    # with DICOM - and the version pin exists precisely to make such mismatches loud
    # rather than mysterious.
    window = if (is.null(window)) NULL
             else if (is.character(window)) window
             else as.numeric(window),
    target_spacing = if (is.null(target_spacing)) NULL else as.numeric(target_spacing),
    channels_from = if (is.null(channels_from)) NULL else as.character(channels_from),
    subdirs = subdirs,
    column_sep = column_sep,
    shuffle = shuffle
  )
}

#' Colormaps and their labels
#'
#' `binary_colormap` is black background, white foreground: the ordinary two-class
#' problem. `voc_colormap` is the twenty-one class palette used by Pascal VOC, kept
#' because datasets and other people's masks still arrive in it.
#'
#' The labels are the matching class names, for figure legends and [mask_coverage()].
#' Position in the list is the class index throughout, counting from 1.
#'
#' @format A list of RGB triples, or a character vector of names.
#' @name colormaps
#' @examples
#' mask_coverage(matrix(c(1, 2, 2, 1), 2), labels = binary_labels)
NULL

#' @rdname colormaps
#' @export
binary_colormap <- list(c(0L, 0L, 0L), c(255L, 255L, 255L))

#' @rdname colormaps
#' @export
binary_labels <- c("background", "object")

#' @rdname colormaps
#' @export
voc_colormap <- list(
  c(0L, 0L, 0L),       c(128L, 0L, 0L),     c(0L, 128L, 0L),     c(128L, 128L, 0L),
  c(0L, 0L, 128L),     c(128L, 0L, 128L),   c(0L, 128L, 128L),   c(128L, 128L, 128L),
  c(64L, 0L, 0L),      c(192L, 0L, 0L),     c(64L, 128L, 0L),    c(192L, 128L, 0L),
  c(64L, 0L, 128L),    c(192L, 0L, 128L),   c(64L, 128L, 128L),  c(192L, 128L, 128L),
  c(0L, 64L, 0L),      c(128L, 64L, 0L),    c(0L, 192L, 0L),     c(128L, 192L, 0L),
  c(0L, 64L, 128L)
)

#' @rdname colormaps
#' @export
voc_labels <- c(
  "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
  "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "potted plant",
  "sheep", "sofa", "train", "tv/monitor"
)

#' The named CT windows
#'
#' The windows radiologists use, as `c(centre, width)` in Hounsfield units. Pass a name to
#' [segmentation_data()] rather than the numbers; this is here for looking them up.
#'
#' Held here rather than fetched from the engine so that looking up a constant does not
#' require starting Python. A test checks the two lists against each other whenever the
#' engine is available, so the copy cannot quietly drift.
#'
#' @return A named list of `c(centre, width)` pairs.
#' @export
#' @examples
#' ct_windows()$lung
#' ct_windows()[["soft_tissue"]]
ct_windows <- function() {
  list(
    brain       = c(centre = 40,   width = 80),
    subdural    = c(centre = 75,   width = 215),
    stroke      = c(centre = 32,   width = 8),
    bone        = c(centre = 400,  width = 1800),
    soft_tissue = c(centre = 40,   width = 400),
    abdomen     = c(centre = 60,   width = 400),
    liver       = c(centre = 30,   width = 150),
    lung        = c(centre = -600, width = 1500),
    mediastinum = c(centre = 50,   width = 350)
  )
}

#' Build an experiment specification
#'
#' Either from arguments, or from a YAML file by passing its path as the only argument.
#' Both produce the same object, and everything downstream treats them identically.
#'
#' The engine validates it, so a mistake is caught here rather than part-way through a
#' training run - and reported against the model that has it, by the name you gave it.
#'
#' @param data A [segmentation_data()] specification, or the path to a YAML file.
#' @param models A list of model specifications, see [models].
#' @param seed Set it for a reproducible run.
#' @param output_dir Where to write a record of the run: the specification, the history,
#'   and anything the run worked out that the specification does not already say - a
#'   detector's fitted anchors, which it cannot be reloaded without. `NULL`, the default,
#'   writes nothing.
#'
#'   Unset means *nothing is written*, not "written to a default place". A field with a
#'   default cannot be told apart from one somebody set to that default, so sending the
#'   default from here would have put files in every user's working directory the moment
#'   the engine learned to write them - which it just did.
#' @param check_paths Verify that the data paths exist. Turn it off to build a
#'   specification on a machine that does not hold the data.
#' @return A `platypus_spec`.
#' @export
#' @examples
#' \dontrun{
#' spec <- platypus_spec(
#'   data = segmentation_data("train/", "valid/", colormap = binary_colormap),
#'   models = list(
#'     u_net("unet", input_shape = c(256, 256)),
#'     linknet("linknet", input_shape = c(256, 256), loss = loss_focal_tversky(alpha = 0.7))
#'   )
#' )
#'
#' # The same thing, from a file
#' spec <- platypus_spec("experiment.yaml")
#' }
platypus_spec <- function(data, models = NULL, seed = NULL,
                          output_dir = NULL, check_paths = TRUE) {
  from_file <- is.character(data) && length(data) == 1L && is.null(models)

  result <- if (from_file) {
    shim()$load_spec(path.expand(data), check_paths = check_paths)
  } else {
    if (is.null(models) || !length(models)) {
      stop("`models` must contain at least one model; see `?models`.", call. = FALSE)
    }
    task <- agreed_task(data, models)
    config <- compact(list(
      # `task` is sent always, not only for detection. The engine defaults a missing one
      # to segmentation, so omitting it would work - and then a reader of the request
      # could not tell which task was meant, which is the thing `task` exists to say.
      task = task,
      data = data, models = models,
      seed = int1(seed), output_dir = output_dir
    ))
    shim()$build_spec(config, check_paths = check_paths)
  }

  if (!isTRUE(result$ok)) abort_engine(result)
  new_spec(result$spec, source = if (from_file) data else NULL)
}

new_spec <- function(py_spec, source = NULL) {
  structure(
    list(py = py_spec, source = source),
    class = "platypus_spec"
  )
}

#' @export
print.platypus_spec <- function(x, ...) {
  info <- shim()$spec_as_dict(x$py)
  models <- info$models
  cat("platypus specification\n")
  if (!is.null(x$source)) cat("  from        :", x$source, "\n")
  cat("  data        :", info$data$train_path, "\n")
  cat("  classes     :", length(info$data$colormap), "\n")
  cat("  models      :", length(models), "\n")
  for (m in models) {
    shape <- paste(unlist(m$input_shape), collapse = "x")
    tiles <- if (!is.null(m$splits)) {
      paste0(", tiled ", paste(unlist(m$splits), collapse = "x"))
    } else ""
    cat(sprintf("    %-14s %-16s %s%s, %s, %d epochs\n",
                m$name, m$architecture, shape, tiles, m$loss$name, m$epochs))
  }
  invisible(x)
}

#' Turn a specification back into plain R data
#'
#' Useful for looking at what the engine made of your arguments, and for writing it out.
#' @param x A `platypus_spec`.
#' @param ... Unused.
#' @return A nested list.
#' @export
as.list.platypus_spec <- function(x, ...) shim()$spec_as_dict(x$py)

#' Drop empty entries so the engine's own defaults apply
#'
#' R keeps a `NULL` in a list rather than dropping it, so an argument left unset would
#' otherwise cross over as an explicit `None` and overwrite whatever default the engine
#' has. Recursive, because specifications nest.
#' @keywords internal
#' @noRd
compact <- function(x) {
  if (!is.list(x)) return(x)
  x <- lapply(x, compact)
  x[!vapply(x, function(v) is.null(v) || (is.list(v) && !length(v)), logical(1))]
}


#' The one task a specification describes, or a refusal naming the disagreement
#'
#' Inferred from the objects rather than asked for: `detection_data()` and `yolo3()` carry
#' it, and a user who chose them has said which they meant twice already.
#'
#' Caught here rather than across the bridge because the engine's refusal would be
#' "data.classes: Extra inputs are not permitted", which describes the symptom. Mixing a
#' detection data block with a segmentation model is not a stray field, it is two
#' intentions in one specification.
#'
#' @noRd
agreed_task <- function(data, models) {
  from_data <- task_of(data)
  from_models <- vapply(models, task_of, character(1))
  all_tasks <- unique(c(from_data, from_models))

  if (length(all_tasks) == 1L) return(all_tasks)

  named <- vapply(seq_along(models), function(i) {
    name <- models[[i]][["name"]]
    sprintf("%s is %s", if (is.null(name)) paste0("models[[", i, "]]") else name,
            from_models[[i]])
  }, character(1))
  stop("a specification describes one task, and this one mixes them: the data is ",
       from_data, " while ", paste(named, collapse = ", "), ".\n",
       "  Masks and boxes are not the same pipeline, so there is nothing to reconcile - ",
       "use `segmentation_data()` with `u_net()` and friends, or `detection_data()` with ",
       "`yolo3()`.", call. = FALSE)
}
