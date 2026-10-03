# Refusing to train on masks the colormap does not describe, from R.
#
# The engine does the refusing. What matters here is that it arrives as a sentence, that
# the two things the message tells you to do next are both reachable from R - which they
# were not when this was written - and that turning the check off works.

test_that("platypus_fit takes check_masks and defaults to on", {
  formals <- names(formals(platypus_fit))
  expect_true("check_masks" %in% formals)
  expect_true(eval(formals(platypus_fit)$check_masks))
})

test_that("mask_report refuses anything that is not a spec", {
  expect_error(mask_report(list(a = 1)), "platypus_spec")
})


# --- against a real engine -------------------------------------------------------------

test_that("a colormap matching no tissue is refused, readably", {
  skip_if_no_mask_check()
  root <- tiny_dataset(n = 6, size = 32)
  spec <- platypus_spec(
    # The masks the fixture writes are white; this asks for VOC red.
    data = segmentation_data(root, root, colormap = list(c(0, 0, 0), c(128, 0, 0))),
    models = list(u_net("m", input_shape = c(32, 32), blocks = 2, filters = 4, epochs = 1))
  )
  expect_error(platypus_fit(spec), "never appears in the training masks")
})

test_that("the refusal names the colour, not only the class number", {
  skip_if_no_mask_check()
  root <- tiny_dataset(n = 6, size = 32)
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = list(c(0, 0, 0), c(128, 0, 0))),
    models = list(u_net("m", input_shape = c(32, 32), blocks = 2, filters = 4, epochs = 1))
  )
  expect_error(platypus_fit(spec), "128")
})

test_that("check_masks = FALSE trains anyway", {
  skip_if_no_mask_check()
  root <- tiny_dataset(n = 6, size = 32)
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = list(c(0, 0, 0), c(128, 0, 0))),
    models = list(u_net("m", input_shape = c(32, 32), blocks = 2, filters = 4, epochs = 1,
                        batch_size = 2))
  )
  fit <- platypus_fit(spec, check_masks = FALSE)
  expect_s3_class(fit, "platypus_fit")
})

test_that("a correct colormap is not refused", {
  skip_if_no_mask_check()
  root <- tiny_dataset(n = 6, size = 32)
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("m", input_shape = c(32, 32), blocks = 2, filters = 4, epochs = 1,
                        batch_size = 2))
  )
  expect_s3_class(platypus_fit(spec), "platypus_fit")
})

test_that("mask_report says what matched and what did not", {
  skip_if_no_mask_check()
  root <- tiny_dataset(n = 6, size = 32)
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("m", input_shape = c(32, 32), blocks = 2, filters = 4, epochs = 1))
  )
  report <- mask_report(spec)
  expect_s3_class(report, "data.frame")
  expect_identical(nrow(report), 1L)
  expect_identical(report$split, "train")
  expect_identical(attr(report, "missing"), integer(0))
  expect_true(all(c(0L, 1L) %in% attr(report, "present")))
  expect_lte(report$unmatched, 0.01)
  expect_identical(report$total_samples, 6L)
})

test_that("mask_report names the class a wrong colormap loses", {
  skip_if_no_mask_check()
  root <- tiny_dataset(n = 6, size = 32)
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = list(c(0, 0, 0), c(128, 0, 0))),
    models = list(u_net("m", input_shape = c(32, 32), blocks = 2, filters = 4, epochs = 1))
  )
  report <- mask_report(spec)
  expect_identical(attr(report, "missing"), 1L)
  expect_identical(report$missing_classes, "1")
  # It reads without training, which is the point of having it.
  expect_gt(report$samples_checked, 0L)
})

test_that("mask_report reads the split it is asked for", {
  skip_if_no_mask_check()
  root <- tiny_dataset(n = 6, size = 32)
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("m", input_shape = c(32, 32), blocks = 2, filters = 4, epochs = 1))
  )
  expect_identical(mask_report(spec, split = "validation")$split, "validation")
})

test_that("the two things the refusal tells you to do are both reachable from R", {
  skip_if_no_mask_check()
  # This is the whole reason mask_report() and the argument exist. The engine's message
  # names `Engine(..., check_masks=False)` and an `inspect_masks` call - neither of which
  # an R user can type. A message that suggests something impossible is worse than one
  # that only says what is wrong, so both have an R spelling and both are exercised here.
  root <- tiny_dataset(n = 6, size = 32)
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = list(c(0, 0, 0), c(128, 0, 0))),
    models = list(u_net("m", input_shape = c(32, 32), blocks = 2, filters = 4, epochs = 1,
                        batch_size = 2))
  )
  expect_error(platypus_fit(spec))                       # refused
  expect_identical(attr(mask_report(spec), "missing"), 1L)   # and inspectable
  expect_s3_class(platypus_fit(spec, check_masks = FALSE), "platypus_fit")  # and overridable
})
