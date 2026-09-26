# Every test that needs Python goes through this.
#
# R CMD check has to pass on a machine with no Python, no network and no patience - that
# is how CRAN runs it, and how a reviewer will see the package. So nothing below starts
# the engine unless it is already there or the environment says it is allowed to.

engine_available <- function() {
  if (!identical(Sys.getenv("PLATYPUS_TEST_ENGINE"), "true")) {
    return(FALSE)
  }
  isTRUE(tryCatch({
    platypus_status()
    reticulate::py_available(initialize = TRUE)
  }, error = function(e) FALSE))
}

skip_if_no_engine <- function() {
  testthat::skip_if_not(
    engine_available(),
    "engine not started (set PLATYPUS_TEST_ENGINE=true to run these)"
  )
}

#' A tiny dataset on disk, laid out the way `nested_dirs` expects
#'
#' Written through the engine's own Python rather than by adding an image-writing
#' dependency to this package: these tests need the engine anyway, so nothing is gained by
#' having two ways to make a PNG.
tiny_dataset <- function(n = 6, size = 32) {
  root <- withr::local_tempdir(.local_envir = parent.frame())
  reticulate::py_run_string(sprintf("
import numpy as np, pathlib
from PIL import Image
root = pathlib.Path(%s)
rng = np.random.default_rng(0)
for i in range(%d):
    sample = root / f'sample_{i:02d}'
    (sample / 'images').mkdir(parents=True, exist_ok=True)
    (sample / 'masks').mkdir(parents=True, exist_ok=True)
    mask = np.zeros((%d, %d), np.uint8)
    top = 4 + (i %% 8)
    mask[top:top + 12, 6:22] = 255
    image = np.clip(mask.astype(np.float32) * 0.7 + rng.random((%d, %d)) * 70, 0, 255)
    Image.fromarray(image.astype(np.uint8)).convert('RGB').save(sample / 'images' / 'a.png')
    Image.fromarray(mask).convert('RGB').save(sample / 'masks' / 'a.png')
", shQuote(root), n, size, size, size, size))
  root
}

tiny_spec <- function(root, ...) {
  platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("tiny", input_shape = c(32, 32), blocks = 2, filters = 4,
                        metrics = list(metric_dice(include_background = FALSE)),
                        epochs = 1, batch_size = 2, ...))
  )
}

#' Does the engine in use know about DICOM?
#'
#' The R package pins an exact pyplatypus, which is what stops a release breaking when the
#' engine drifts - but it also means a feature can exist here before the engine carrying it
#' is published. Until then CI runs against the released version, and these tests have to
#' notice that rather than fail over it.
engine_reads_dicom <- function() {
  if (!engine_available()) return(FALSE)
  isTRUE(tryCatch({
    length(platypus:::shim()$window_presets()) > 0
  }, error = function(e) FALSE))
}

skip_if_no_dicom <- function() {
  skip_if_no_engine()
  testthat::skip_if_not(
    engine_reads_dicom(),
    "the engine in use predates DICOM support"
  )
}


#' Does the engine in use know how to split a dataset and score cases?
#'
#' Same reasoning as `engine_reads_dicom()`: the pin means this package can describe a
#' feature before the published engine carries it, and CI runs against the published one.
#' Asking the engine what it has beats assuming.
engine_splits_data <- function() {
  if (!engine_available()) return(FALSE)
  isTRUE(tryCatch({
    engine <- platypus:::engine()
    reticulate::py_has_attr(engine, "split_dataset") &&
      reticulate::py_has_attr(engine, "summarise_cases")
  }, error = function(e) FALSE))
}

skip_if_no_splits <- function() {
  skip_if_no_engine()
  testthat::skip_if_not(
    engine_splits_data(),
    "the engine in use predates dataset splitting"
  )
}


#' A dataset laid out as scans of several patients
#'
#' Keys look like `patient03_slice001`, which is the shape a group pattern is written for.
#' @param envir The frame the directory should outlive. Training reads the files once, but
#'   `evaluate_cases()` reads them again afterwards, so a dataset tied to a helper's own
#'   frame vanishes between the two - which is how this argument came to exist.
tiny_patient_dataset <- function(patients = 8, slices = 2, size = 32,
                                 envir = parent.frame()) {
  root <- withr::local_tempdir(.local_envir = envir)
  reticulate::py_run_string(sprintf("
import numpy as np, pathlib
from PIL import Image
root = pathlib.Path(%s)
rng = np.random.default_rng(0)
for p in range(%d):
    for s in range(%d):
        sample = root / f'patient{p:02d}_slice{s:03d}'
        (sample / 'images').mkdir(parents=True, exist_ok=True)
        (sample / 'masks').mkdir(parents=True, exist_ok=True)
        mask = np.zeros((%d, %d), np.uint8)
        top = 4 + p
        mask[top:top + 12, 6:22] = 255
        image = np.clip(mask.astype(np.float32) * 0.7 + rng.random((%d, %d)) * 70, 0, 255)
        Image.fromarray(image.astype(np.uint8)).convert('RGB').save(sample / 'images' / 'a.png')
        Image.fromarray(mask).convert('RGB').save(sample / 'masks' / 'a.png')
", shQuote(root), patients, slices, size, size, size, size))
  root
}


#' Can the engine in use read volumes?
engine_reads_volumes <- function() {
  if (!engine_available()) return(FALSE)
  isTRUE(tryCatch(platypus:::shim()$volume_support(), error = function(e) FALSE))
}

skip_if_no_volumes <- function() {
  skip_if_no_engine()
  testthat::skip_if_not(
    engine_reads_volumes(),
    "the engine in use predates volume support"
  )
}

#' A dataset of NIfTI volumes with label-map masks
#'
#' Deliberately tiny. A 3D fixture that takes a minute gets skipped, and a skipped test
#' proves nothing.
tiny_volume_dataset <- function(cases = 4, shape = c(8, 8, 4), spacing = c(2, 2, 5),
                                envir = parent.frame()) {
  root <- withr::local_tempdir(.local_envir = envir)
  reticulate::py_run_string(sprintf("
import numpy as np, pathlib, nibabel as nib
root = pathlib.Path(%s)
shape = (%d, %d, %d)
affine = np.diag([%f, %f, %f, 1.0])
for n in range(%d):
    sample = root / f'case_{n:02d}'
    (sample / 'images').mkdir(parents=True, exist_ok=True)
    (sample / 'masks').mkdir(parents=True, exist_ok=True)
    labels = np.zeros(shape, np.float32)
    labels[2:6, 2:6, 1:3] = 1
    scan = np.where(labels > 0, 40.0, -1000.0).astype(np.float32)
    nib.save(nib.Nifti1Image(scan, affine), str(sample / 'images' / 'ct.nii.gz'))
    nib.save(nib.Nifti1Image(labels, affine), str(sample / 'masks' / 'seg.nii.gz'))
", shQuote(root), shape[1], shape[2], shape[3],
   spacing[1], spacing[2], spacing[3], cases))
  root
}


#' Can the engine in use assemble a DICOM series?
engine_reads_series <- function() {
  if (!engine_available()) return(FALSE)
  isTRUE(tryCatch(platypus:::shim()$series_support(), error = function(e) FALSE))
}

skip_if_no_series <- function() {
  skip_if_no_engine()
  testthat::skip_if_not(
    engine_reads_series(),
    "the engine in use predates DICOM series support"
  )
}

#' A folder of DICOM slices, written through the engine's pydicom
#'
#' `positions` are millimetres along the slice normal, so a caller can leave a gap or repeat a
#' position to build the broken cases the report exists to find. File names deliberately sort
#' differently from the anatomy.
#' @param prefix Distinguishes two series written into one directory, which is the state an
#'   archive export arrives in and what the report exists to catch. Without it the second call
#'   writes the same filenames and silently replaces the first.
tiny_series <- function(directory, positions = c(0, 2.5, 5, 7.5), uid = NULL,
                        prefix = "IM") {
  reticulate::py_run_string(sprintf("
import pathlib, pydicom
from pydicom.data import get_testdata_file
directory = pathlib.Path(%s)
directory.mkdir(parents=True, exist_ok=True)
positions = [%s]
uid = %s or pydicom.uid.generate_uid()
prefix = %s
for index, position in enumerate(positions):
    dataset = pydicom.dcmread(get_testdata_file('CT_small.dcm'))
    dataset.SeriesInstanceUID = uid
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0.0, 0.0, float(position)]
    dataset.PixelSpacing = [0.8, 0.8]
    dataset.SliceThickness = 2.5
    dataset.InstanceNumber = index + 1
    dataset.save_as(directory / f'{prefix}{(len(positions) - index) * 7 %% 100:02d}.dcm')
", shQuote(directory), paste(positions, collapse = ", "),
   if (is.null(uid)) "None" else shQuote(uid), shQuote(prefix)))
  directory
}
