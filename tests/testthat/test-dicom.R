test_that("the windows are available without starting Python", {
  # Looking up a constant should not cost an interpreter, and `R CMD check` runs examples
  # on machines that have none.
  windows <- ct_windows()
  expect_true(all(c("lung", "soft_tissue", "bone", "brain") %in% names(windows)))
  expect_identical(names(windows$lung), c("centre", "width"))
  expect_identical(unname(windows$lung), c(-600, 1500))
})

test_that("the copy kept here matches the engine's", {
  # Two lists of the same constants drift, and the one that drifts is always the one
  # nobody is testing. This is that test.
  skip_if_no_dicom()
  theirs <- platypus:::shim()$window_presets()
  ours <- ct_windows()
  expect_setequal(names(ours), names(theirs))
  for (name in names(ours)) {
    expect_equal(unname(ours[[name]]), as.numeric(unlist(theirs[[name]])),
                 info = paste("window", name))
  }
})

test_that("a window can be named or given as numbers", {
  skip_if_no_dicom()
  for (window in list("soft_tissue", c(40, 400), "auto", "full")) {
    spec <- platypus_spec(
      data = segmentation_data(tempdir(), tempdir(), binary_colormap,
                               dicom_window = window),
      models = list(u_net("u", input_shape = c(64, 64)))
    )
    expect_s3_class(spec, "platypus_spec")
  }
})

test_that("a misspelled window is answered with the real ones", {
  skip_if_no_dicom()
  error <- tryCatch(
    platypus_spec(
      data = segmentation_data(tempdir(), tempdir(), binary_colormap,
                               dicom_window = "sof_tissue"),
      models = list(u_net("u", input_shape = c(64, 64)))
    ),
    error = function(e) e
  )
  expect_s3_class(error, "platypus_error")
  expect_match(conditionMessage(error), "soft_tissue")
  expect_match(conditionMessage(error), "data\\$dicom_window")
})

test_that("a window of no width is refused", {
  skip_if_no_dicom()
  expect_error(
    platypus_spec(
      data = segmentation_data(tempdir(), tempdir(), binary_colormap,
                               dicom_window = c(40, 0)),
      models = list(u_net("u", input_shape = c(64, 64)))
    ),
    "positive"
  )
})

test_that("the window survives the crossing intact", {
  # A setting nothing acts on is worse than no setting: it reads as a decision the user
  # made and the software ignored.
  skip_if_no_dicom()
  spec <- platypus_spec(
    data = segmentation_data(tempdir(), tempdir(), binary_colormap,
                             dicom_window = c(35, 350)),
    models = list(u_net("u", input_shape = c(64, 64)))
  )
  expect_equal(unlist(as.list(spec)$data$dicom_window), c(35, 350))
})

test_that("the default leaves the file's own window in charge", {
  skip_if_no_dicom()
  spec <- platypus_spec(
    data = segmentation_data(tempdir(), tempdir(), binary_colormap),
    models = list(u_net("u", input_shape = c(64, 64)))
  )
  expect_identical(as.list(spec)$data$dicom_window, "auto")
})
