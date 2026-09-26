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
