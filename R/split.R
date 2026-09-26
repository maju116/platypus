# Splitting a dataset before training on it.
#
# The engine does the work; this is the surface. What matters here is that the result drops
# straight into `segmentation_data()`, because the whole point is that getting from a folder
# of images to a trained model is a handful of lines.

#' Split a dataset into training, validation and test sets
#'
#' Divides one directory of samples into three sets and writes each as a CSV the
#' specification can point at. Nothing is copied: the files stay where they are and the
#' CSVs list them, which matters when the data is large or read-only.
#'
#' @section Why `group_by` exists:
#'
#' Splitting per file is the obvious thing and, for medical images, the wrong one. A scan
#' is many slices of one patient, so dividing slices at random puts the same anatomy in
#' training and in validation at once. The model then meets the neighbouring slice of what
#' it is being tested on, validation measures memory rather than generalisation, and Dice
#' comes out several points too high with nothing in the result to say so.
#'
#' `group_by` names the unit that must not be split - usually a patient, sometimes a study,
#' a scanner or a site. Every sample sharing a group lands in exactly one set.
#'
#' @section Patterns are Python regular expressions:
#'
#' `group_by` is passed to the engine exactly as written, and read there as a Python
#' regular expression. In practice the dialects agree on everything you are likely to need
#' - `^`, `$`, `\\d`, `[A-Z]`, `+`, `*`, `()` - and R's own quoting rules still apply, so a
#' backslash is doubled: `"^(patient\\\\d+)_"`. It is not translated on the way through,
#' deliberately: a translation that quietly got it wrong would produce a working split with
#' the wrong groups in it, which is the failure this argument exists to prevent.
#'
#' A pattern that matches no sample is an error, never a fallback to one-group-per-sample.
#'
#' @param root The directory holding the samples, laid out as `mode` describes.
#' @param out_dir Where to write `train.csv`, `validation.csv` and `test.csv`.
#' @param group_by A pattern picking the group out of a sample's name; see above. `NULL`
#'   treats every sample as its own group, which is right when one sample is one patient.
#' @param fractions Two or three shares that add up to 1: train, validation and optionally
#'   test. Measured in samples, not in groups, so patients with very different numbers of
#'   slices do not skew them.
#' @param seed Makes the split reproducible. The same inputs and seed give the same split
#'   on any machine and in any file order - without that, two runs are not comparable and
#'   the reason for a difference is impossible to find.
#' @param mode `"nested_dirs"` for one directory per sample, `"config_file"` for a CSV.
#' @param subdirs The image and mask subdirectory names, for `nested_dirs`.
#' @param column_sep Separator between several paths in one CSV cell.
#' @param strict Treat an incomplete sample as an error rather than skipping it.
#' @param relative Write paths relative to the CSV where possible, so the dataset and its
#'   split can be moved together.
#' @return A `platypus_split`: the three paths, plus how many samples and groups each set
#'   received. Pass it straight to [segmentation_data()].
#' @seealso [evaluate_cases()], which reports a score per case rather than one per set.
#' @examples
#' \dontrun{
#' split <- platypus_split(
#'   "scans", "splits",
#'   group_by = "^(patient\\\\d+)_",
#'   fractions = c(0.7, 0.15, 0.15)
#' )
#' split
#'
#' spec <- platypus_spec(
#'   data = segmentation_data(split, colormap = binary_colormap),
#'   models = list(u_net("unet", input_shape = c(256, 256)))
#' )
#' }
#' @export
platypus_split <- function(root, out_dir, group_by = NULL,
                           fractions = c(0.7, 0.15, 0.15), seed = 0,
                           mode = c("nested_dirs", "config_file"),
                           subdirs = c("images", "masks"), column_sep = ";",
                           strict = TRUE, relative = TRUE) {
  mode <- match.arg(mode)
  if (!is.numeric(fractions) || length(fractions) < 2 || length(fractions) > 3) {
    stop("`fractions` must be two or three numbers (train, validation[, test]).",
         call. = FALSE)
  }

  result <- shim()$split_data(
    root = path.expand(root), out_dir = path.expand(out_dir), mode = mode,
    subdirs = as.character(subdirs), column_sep = column_sep,
    fractions = as.numeric(fractions), group_by = group_by, seed = as.integer(seed),
    strict = strict, relative = relative
  )
  if (!isTRUE(result$ok)) abort_engine(result)

  structure(
    list(
      train_path = result$paths$train,
      validation_path = result$paths$validation,
      test_path = result$paths$test,
      samples = unlist(result$samples),
      groups = unlist(result$groups),
      skipped = result$skipped,
      group_by = group_by,
      seed = as.integer(seed)
    ),
    class = "platypus_split"
  )
}

#' @export
print.platypus_split <- function(x, ...) {
  cat("platypus split\n")
  grouped <- !is.null(x$group_by)
  for (name in c("train", "validation", "test")) {
    path <- x[[paste0(name, "_path")]]
    if (is.null(path)) next
    samples <- x$samples[[name]]
    groups <- x$groups[[name]]
    line <- sprintf("  %-11s %4d samples", name, samples)
    if (grouped) line <- paste0(line, sprintf(" in %d groups", groups))
    cat(line, "\n", sep = "")
  }
  if (grouped) {
    cat("  grouped by '", x$group_by, "'; no group is in two sets\n", sep = "")
  } else {
    cat("  every sample its own group\n")
  }
  if (isTRUE(x$skipped > 0)) {
    cat("  ", x$skipped, " sample(s) skipped as incomplete\n", sep = "")
  }
  cat("  seed", x$seed, "- the same inputs give this split again\n")
  invisible(x)
}

#' The files in one part of a split
#'
#' Reads the CSV that [platypus_split()] wrote and hands back the paths ready to use.
#'
#' The reason this exists rather than `read.csv()`: the CSVs store paths **relative to
#' themselves**, so that a dataset and its split can be moved or mounted elsewhere together.
#' Read directly, those paths do not open from wherever your session happens to be - which is a
#' trap worth removing rather than documenting.
#'
#' @param split A [platypus_split()].
#' @param which `"train"`, `"validation"` or `"test"`.
#' @return A data frame with `key`, `group`, `images` and `masks`, the paths made absolute.
#'   `images` and `masks` hold one path per sample, or several separated by the split's
#'   separator when a sample has several files.
#' @examples
#' \dontrun{
#' split <- platypus_split("scans", "splits", group_by = "^(patient\\\\d+)_")
#' validation <- split_files(split, "validation")
#' images <- read_images(validation$images, size = c(32, 32, 32), channels = 1)
#' }
#' @export
split_files <- function(split, which = c("train", "validation", "test")) {
  if (!inherits(split, "platypus_split")) {
    stop("`split` must come from `platypus_split()`.", call. = FALSE)
  }
  which <- match.arg(which)
  path <- split[[paste0(which, "_path")]]
  if (is.null(path)) {
    stop("this split has no '", which, "' part.", call. = FALSE)
  }

  table <- utils::read.csv(path, stringsAsFactors = FALSE)
  base <- dirname(path)
  absolute <- function(cell) {
    vapply(strsplit(cell, ";", fixed = TRUE), function(pieces) {
      pieces <- trimws(pieces)
      resolved <- ifelse(startsWith(pieces, "/") | grepl("^[A-Za-z]:", pieces),
                         pieces, file.path(base, pieces))
      paste(resolved, collapse = ";")
    }, character(1))
  }

  table$images <- absolute(table$images)
  if ("masks" %in% names(table)) table$masks <- absolute(table$masks)
  table
}
