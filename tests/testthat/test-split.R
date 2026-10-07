test_that("`fractions` is checked before Python is ever started", {
  # Cheap arguments are checked in R so a typo costs nothing. Starting an interpreter to be
  # told that three numbers were needed is a poor trade.
  expect_error(split_dataset("x", "y", fractions = 0.7), "two or three numbers")
  expect_error(split_dataset("x", "y", fractions = c(0.5, 0.2, 0.2, 0.1)),
               "two or three numbers")
})

test_that("a split divides a dataset and says what it did", {
  skip_if_no_splits()
  root <- tiny_patient_dataset()
  out <- withr::local_tempdir()

  split <- split_dataset(root, out, group_by = "^(patient\\d+)_",
                         fractions = c(0.5, 0.25, 0.25), seed = 0)

  expect_s3_class(split, "platypus_split")
  expect_equal(sum(split$samples), 16)
  expect_equal(sum(split$groups), 8)
  expect_true(all(file.exists(unlist(split[c("train_path", "validation_path",
                                             "test_path")]))))
})

test_that("no patient lands in two sets", {
  # The reason the whole function exists. Checked from the CSVs rather than from the
  # engine's own bookkeeping, because the CSVs are what training will actually read.
  skip_if_no_splits()
  root <- tiny_patient_dataset()
  out <- withr::local_tempdir()

  split <- split_dataset(root, out, group_by = "^(patient\\d+)_", seed = 3)
  groups <- lapply(c("train_path", "validation_path", "test_path"), function(field) {
    unique(utils::read.csv(split[[field]])$group)
  })

  expect_length(intersect(groups[[1]], groups[[2]]), 0)
  expect_length(intersect(groups[[1]], groups[[3]]), 0)
  expect_length(intersect(groups[[2]], groups[[3]]), 0)
})

test_that("the same seed gives the same split", {
  skip_if_no_splits()
  root <- tiny_patient_dataset()

  first <- split_dataset(root, withr::local_tempdir(), group_by = "^(patient\\d+)_",
                         seed = 11)
  again <- split_dataset(root, withr::local_tempdir(), group_by = "^(patient\\d+)_",
                         seed = 11)

  validation <- function(split) sort(utils::read.csv(split$validation_path)$key)
  expect_identical(validation(first), validation(again))
})

test_that("a pattern matching nothing is refused, not silently ignored", {
  # If this ever becomes a warning, the package starts producing inflated scores quietly.
  skip_if_no_splits()
  expect_error(
    split_dataset(tiny_patient_dataset(), withr::local_tempdir(),
                  group_by = "^(subject\\d+)_"),
    "does not match"
  )
})

test_that("a split goes straight into segmentation_data()", {
  skip_if_no_splits()
  split <- split_dataset(tiny_patient_dataset(), withr::local_tempdir(),
                         group_by = "^(patient\\d+)_", fractions = c(0.5, 0.5))

  data <- segmentation_data(split, colormap = binary_colormap)

  expect_identical(data$mode, "config_file")
  expect_identical(data$train_path, split$train_path)
  expect_identical(data$validation_path, split$validation_path)
})

test_that("passing a split and a validation path at once is caught", {
  skip_if_no_splits()
  split <- split_dataset(tiny_patient_dataset(), withr::local_tempdir(),
                         group_by = "^(patient\\d+)_", fractions = c(0.5, 0.5))
  expect_error(segmentation_data(split, "somewhere", colormap = binary_colormap),
               "already a platypus_split")
})

test_that("a split trains, and the printed summary is readable", {
  skip_if_no_splits()
  split <- split_dataset(tiny_patient_dataset(), withr::local_tempdir(),
                         group_by = "^(patient\\d+)_", fractions = c(0.5, 0.5))
  spec <- platypus_spec(
    data = segmentation_data(split, colormap = binary_colormap),
    models = list(u_net("tiny", input_shape = c(32, 32), blocks = 2, filters = 4,
                        metrics = list(metric_dice(include_background = FALSE)),
                        epochs = 1, batch_size = 2))
  )
  fit <- platypus_fit(spec, num_workers = 0)

  expect_s3_class(fit, "platypus_fit")
  expect_output(print(split), "no group is in two sets")
})

# --- validation = FALSE --------------------------------------------------------------------

test_that("validation = FALSE is the third answer and lives in the same argument", {
  data <- segmentation_data("a", validation = FALSE, colormap = binary_colormap)
  expect_false(data$validation)
  expect_null(data$validation_path)
})

test_that("it survives compact(), which would otherwise silently restore the requirement", {
  # FALSE is not NULL, so `compact()` keeps it - but that is worth a test rather than a
  # reading, because dropping it would leave the engine demanding a validation set from a
  # caller who had just said there is none.
  data <- segmentation_data("a", validation = FALSE, colormap = binary_colormap)
  expect_false(platypus:::compact(data)$validation)
})

test_that("leaving both unset is still an error, and the message names the third answer", {
  expect_error(segmentation_data("a", colormap = binary_colormap),
               "validation = FALSE")
})

test_that("FALSE and a split together are refused: one question, one answer", {
  expect_error(
    segmentation_data("a", validation = FALSE,
                      split = list(fractions = c(0.8, 0.2), group_by = NULL),
                      colormap = binary_colormap),
    "Give one or the other"
  )
})

test_that("a detector can say it too", {
  data <- detection_data("a", validation = FALSE, classes = c("x", "y"))
  expect_false(data$validation)
  expect_null(data$validation_path)
  expect_error(detection_data("a", classes = c("x", "y")), "validation = FALSE")
})

test_that("the engine accepts a run with no validation and reports no val_ column", {
  skip_if_no_optional_validation()
  root <- tiny_dataset(n = 6)

  spec <- platypus_spec(
    data = segmentation_data(root, validation = FALSE, colormap = binary_colormap),
    models = list(u_net("u", input_shape = c(32, 32), blocks = 2, filters = 4,
                        epochs = 1, batch_size = 2)),
    seed = 1
  )
  fit <- platypus_fit(spec, device = "cpu")
  history <- training_history(fit)
  expect_true("train_loss" %in% names(history))
  expect_length(grep("^val_", names(history)), 0)
})

test_that("asking for the validation split says validation = FALSE, not 'no masks'", {
  skip_if_no_optional_validation()
  root <- tiny_dataset(n = 6)

  spec <- platypus_spec(
    data = segmentation_data(root, validation = FALSE, colormap = binary_colormap),
    models = list(u_net("u", input_shape = c(32, 32), blocks = 2, filters = 4,
                        epochs = 1, batch_size = 2)),
    seed = 1
  )
  fit <- platypus_fit(spec, device = "cpu")
  # The two are different and the wrong one sends somebody looking for files they never
  # wrote. `evaluate()` defaults to the validation split, so it is the first thing met.
  expect_error(evaluate(fit), "validation: false")
})
