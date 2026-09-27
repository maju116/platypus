# Volumes: the parts that are not just "the same thing with one more axis".
#
# Two of them. A volume mask has to be written with the geometry of the scan it came from,
# or it cannot be laid over that scan by anything. And a segmented structure is reported in
# millilitres, not in voxels - the voxel count is meaningless outside the scanner that
# produced it, and converting needs the spacing, which is why the spacing is carried.

#' What is in a volume file
#'
#' Shape and voxel spacing, read through the engine so the numbers are the ones training
#' would use - including the canonical reorientation, which changes what "the first axis"
#' means.
#'
#' @param paths One or more paths to NIfTI files.
#' @return A data frame with one row per file: `path`, `spacing_1/2/3` in millimetres,
#'   `shape_1/2/3` in voxels, and `voxel_ml`, the volume of one voxel in millilitres.
#' @seealso [mask_volume()], which turns a segmentation into millilitres.
#' @examples
#' \dontrun{
#' volume_info("scans/case_01/images/ct.nii.gz")
#' }
#' @export
volume_info <- function(paths) {
  result <- shim()$volume_info(as.list(as.character(paths)))
  if (!isTRUE(result$ok)) abort_engine(result)

  rows <- lapply(result$volumes, function(row) {
    spacing <- unlist(row$spacing)
    shape <- unlist(row$shape)
    data.frame(
      path = row$path,
      spacing_1 = spacing[[1]], spacing_2 = spacing[[2]], spacing_3 = spacing[[3]],
      shape_1 = shape[[1]], shape_2 = shape[[2]], shape_3 = shape[[3]],
      voxel_ml = row$voxel_ml,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

#' How much of a volume a class occupies, in millilitres
#'
#' A voxel count answers nothing on its own: the same tumour is 4000 voxels on one scanner
#' and 12000 on another, and neither number goes in a report. Multiplied by the volume of a
#' voxel it becomes millilitres, which is comparable between scans, between patients and
#' against whatever the radiologist measured by hand.
#'
#' @param mask A mask of class indices, as [predict()] returns - one volume, not a stack.
#' @param spacing Voxel size in millimetres, length 3. [volume_info()] reports it; so does
#'   the `spacing_1/2/3` columns of its output.
#' @param class Which class to measure, numbered from 1 as the colormap and `labels` are.
#'   Defaults to 2, which is the foreground in a binary segmentation.
#' @return The volume in millilitres.
#' @examples
#' mask <- array(1L, dim = c(10, 10, 4))
#' mask[3:6, 3:6, 2:3] <- 2L
#' # 32 voxels of 2 x 2 x 5 mm: 32 * 20 mm^3 = 640 mm^3 = 0.64 ml
#' mask_volume(mask, spacing = c(2, 2, 5))
#' @export
mask_volume <- function(mask, spacing, class = 2L) {
  if (length(spacing) != 3L || anyNA(spacing) || any(spacing <= 0)) {
    stop("`spacing` must be three positive numbers, in millimetres.", call. = FALSE)
  }
  if (length(dim(mask)) != 3L) {
    stop("`mask` must be one volume with three dimensions; got ",
         if (is.null(dim(mask))) "a vector" else paste(dim(mask), collapse = " x "),
         ". For a stack, apply this to one volume at a time.", call. = FALSE)
  }
  voxels <- sum(mask == as.integer(class))
  voxels * prod(as.numeric(spacing)) / 1000
}

#' One volume per element, whatever form the masks arrived in
#'
#' A list from `predict(space = "source")` passes through untouched, which is the point of that
#' argument existing: its members are already on their scans' grids and need not agree with each
#' other. An array is unstacked instead, and probabilities are collapsed.
#' @keywords internal
#' @noRd
as_volume_list <- function(masks) {
  if (is.list(masks)) {
    volumes <- lapply(masks, function(one) {
      dims <- length(dim(one))
      if (dims == 4L) {
        # (d, h, w, class): probabilities for one volume.
        apply(one, seq_len(3L), which.max)
      } else if (dims == 3L) {
        one
      } else {
        stop("a mask in the list has ", dims, " dimensions; expected 3 (class indices) or ",
             "4 (probabilities for one volume).", call. = FALSE)
      }
    })
    return(volumes)
  }

  dimensions <- length(dim(masks))
  if (dimensions == 5L) {
    # Probabilities: (n, d, h, w, class). Collapse as predict(type = "class") would.
    masks <- apply(masks, seq_len(4L), which.max)
  } else if (dimensions == 3L) {
    masks <- array(masks, dim = c(1L, dim(masks)))
  } else if (dimensions != 4L) {
    stop("`masks` has ", dimensions, " dimensions; expected 3 (one volume), 4 (a stack) ",
         "or 5 (a stack of probabilities).", call. = FALSE)
  }
  # Four dimensions is a stack of class-index volumes and nothing else. One volume's
  # probabilities have exactly the same rank, and guessing between them by looking at the
  # sizes is how a silent misinterpretation gets into a package: it worked here for a stack
  # of two and would have written one wrong file for a stack of four.
  lapply(seq_len(dim(masks)[1]), function(i) masks[i, , , , drop = TRUE])
}

#' Save predicted volumes as NIfTI
#'
#' Writes each mask as a label-map NIfTI carrying the geometry of the scan it was predicted
#' from. That is the whole reason this is not `save_masks()` with a different extension: a
#' mask array without an affine cannot be laid over its scan by any viewer, any registration
#' tool or any volume calculation - it either lands in the wrong place or is refused.
#'
#' Carrying the affine means the mask has to be on the same grid as the scan, which is what
#' `predict(space = "source")` is for: it maps each prediction back out of the model's grid onto
#' the scan's. Without it, a resampled or resized prediction will be refused here rather than
#' written somewhere wrong.
#'
#' @param masks Masks of class indices, as [predict()] returns.
#'
#'   A **list** of volumes is what `predict(space = "source")` gives, and the form to prefer:
#'   each mask is already on the grid of the scan it was computed from, so the geometry matches
#'   by construction and the scans need not be the same size as each other.
#'
#'   An array is also accepted: one volume as `depth x height x width`, a stack as
#'   `volume x depth x height x width`, probabilities as `volume x depth x height x width x class`
#'   which are collapsed to the most likely class. A single volume's probabilities have the same
#'   rank as a stack of masks and are not guessed at - collapse them yourself, or keep the volume
#'   axis.
#' @param dir Where to write. Created if it does not exist.
#' @param reference The scan each mask was predicted from, in the same order. Its affine is
#'   what the mask is written with, so this is required rather than optional.
#' @param names File names, without extension. Taken from `reference` when unset, which works
#'   when the scans have distinct filenames. They often do not - one directory per case, the same
#'   `ct.nii.gz` inside each - and writing every mask to one name would silently keep only the
#'   last, so that case is an error asking for this argument. `split_files(split, "validation")$key`
#'   is usually the answer.
#' @param suffix Appended to each name, for telling two models' output apart.
#' @return The paths written, invisibly.
#' @seealso [save_masks()] for 2D masks as pictures.
#' @examples
#' \dontrun{
#' validation <- split_files(split, "validation")
#' masks <- predict(fit, split = "validation", space = "source")
#' save_volumes(masks, "predictions", reference = validation$images)
#' }
#' @export
save_volumes <- function(masks, dir, reference, names = NULL, suffix = "") {
  volumes <- as_volume_list(masks)

  count <- length(volumes)
  reference <- as.character(reference)
  if (length(reference) != count) {
    stop("`reference` has ", length(reference), " entries but there are ", count,
         " masks. Each mask is written with the geometry of the scan it came from, so ",
         "they have to line up.", call. = FALSE)
  }

  names <- if (is.null(names)) {
    sub("[.]nii([.]gz)?$", "", basename(reference), ignore.case = TRUE)
  } else {
    as.character(names)
  }
  if (length(names) != count) {
    stop("`names` has ", length(names), " entries but there are ", count, " masks.",
         call. = FALSE)
  }
  if (anyDuplicated(names)) {
    # Datasets that keep one directory per case usually name the file inside it the same way -
    # `case_01/images/ct.nii.gz` - so basenames collide and every mask would be written to one
    # path, each overwriting the last. Silently losing five of six masks is the worst outcome
    # available here, and the fix is one argument.
    repeated <- unique(names[duplicated(names)])
    stop("the masks would be written to the same name: ",
         paste(utils::head(repeated, 3), collapse = ", "),
         if (length(repeated) > 3) ", ..." else "",
         ". Names come from the reference files, and one directory per case usually means ",
         "the same filename in each. Pass `names` - `split_files(split, \"validation\")$key` ",
         "is what you want.", call. = FALSE)
  }

  paths <- file.path(dir, paste0(names, suffix, ".nii.gz"))

  result <- shim()$write_volumes(volumes, as.list(paths), as.list(reference))
  if (!isTRUE(result$ok)) abort_engine(result)
  invisible(unlist(result$paths))
}

#' Check folders of DICOM slices before training on them
#'
#' One row per directory, saying whether it is a usable series and what is wrong with it if it
#' is not. Problems are reported as a column rather than raised, because a hundred cases out
#' of an archive will contain a few with a missing slice, two series in one folder, or
#' duplicated files - and stopping at the first turns an afternoon's work into a week's.
#'
#' Reads no pixels. Every check runs on the DICOM headers, so this is usable on a hundred
#' gigabytes of data and is the first thing worth running on a new dataset.
#'
#' @section What it catches:
#'
#' A **missing slice**, which does not make a volume shorter - it puts everything past the gap
#' in the wrong place, and no metric would show it. **Two series in one folder**, which is the
#' normal state of an archive export: a scout, a reconstruction and a phase sitting together,
#' which stacked would interleave two anatomies. **Duplicated slices**, from a copy or from
#' several echoes mixed together. Slices of different **size**, **angle** or **spacing**.
#'
#' It also reports how the slices were ordered. `"position"` means the geometry decided, which
#' is what you want. `"instance_number"` means the files carry no positions and the order came
#' from a number that is only required to be unique - worth knowing before trusting the result.
#'
#' @param paths Directories, each holding the slices of one series.
#' @return A data frame with one row per directory: `path`, `ok`, `slices`, `sorted_by`,
#'   `spacing_1/2/3`, `series_uid` and `problem`.
#' @seealso [volume_info()] for NIfTI files, [platypus_split()] for dividing the cases up.
#' @examples
#' \dontrun{
#' cases <- list.dirs("dicom_export", recursive = FALSE)
#' report <- series_report(cases)
#'
#' report[!report$ok, c("path", "problem")]     # the ones to deal with first
#' table(report$sorted_by)
#' }
#' @export
series_report <- function(paths) {
  result <- shim()$series_report(as.list(as.character(paths)))
  if (!isTRUE(result$ok)) abort_engine(result)

  rows <- lapply(result$series, function(row) {
    spacing <- if (is.null(row$spacing)) rep(NA_real_, 3) else unlist(row$spacing)
    data.frame(
      path = row$path,
      ok = isTRUE(row$ok),
      slices = as.integer(row$slices),
      sorted_by = row$sorted_by %||% NA_character_,
      spacing_1 = spacing[[1]], spacing_2 = spacing[[2]], spacing_3 = spacing[[3]],
      series_uid = row$series_uid %||% NA_character_,
      problem = row$problem %||% NA_character_,
      stringsAsFactors = FALSE
    )
  })
  frame <- do.call(rbind, rows)
  structure(frame, class = c("platypus_series_report", class(frame)))
}

#' @export
print.platypus_series_report <- function(x, ...) {
  usable <- sum(x$ok)
  cat("platypus series report: ", usable, " of ", nrow(x), " usable\n", sep = "")
  if (usable > 0) {
    ordered <- table(x$sorted_by[x$ok])
    cat("  ordered by: ",
        paste(names(ordered), ordered, sep = " x ", collapse = ", "), "\n", sep = "")
  }
  broken <- x[!x$ok, , drop = FALSE]
  if (nrow(broken)) {
    cat("  not usable:\n")
    for (i in seq_len(min(5L, nrow(broken)))) {
      first_line <- strsplit(broken$problem[[i]], "[.] ")[[1]][[1]]
      cat("   - ", basename(broken$path[[i]]), ": ", first_line, "\n", sep = "")
    }
    if (nrow(broken) > 5L) {
      cat("   ... and ", nrow(broken) - 5L, " more; see the `problem` column.\n", sep = "")
    }
  }
  invisible(x)
}
