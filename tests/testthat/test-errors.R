test_that("python paths are rewritten as the R that produced them", {
  # Zero-based indices in an R error message are a small cruelty, and `.` is not how one
  # reaches into a list.
  expect_identical(platypus:::r_path("models[0].loss"), "models[[1]]$loss")
  expect_identical(platypus:::r_path("data.colormap"), "data$colormap")
  expect_identical(platypus:::r_path("models[2].augmentation[0].name"),
                   "models[[3]]$augmentation[[1]]$name")
  expect_identical(platypus:::r_path(""), "(top level)")
})

test_that("every problem is reported, not only the first", {
  skip_if_no_engine()
  error <- tryCatch(
    platypus_spec(
      data = segmentation_data(tempdir(), tempdir(),
                               colormap = list(c(0, 0, 0), c(300, 0, 0))),
      models = list(u_net("u", input_shape = c(255, 255), dropout = 2,
                          loss = list(name = "focaal")))
    ),
    error = function(e) e
  )
  expect_s3_class(error, "platypus_error")
  expect_gte(nrow(error$problems), 3L)
})

test_that("a misspelled loss is answered with the list of real ones", {
  skip_if_no_engine()
  error <- tryCatch(
    platypus_spec(
      data = segmentation_data(tempdir(), tempdir(), colormap = binary_colormap),
      models = list(u_net("u", input_shape = c(64, 64), loss = list(name = "focaal")))
    ),
    error = function(e) e
  )
  expect_match(conditionMessage(error), "focal_tversky")
})

test_that("a misspelled augmentation gets a suggestion", {
  skip_if_no_engine()
  error <- tryCatch(
    platypus_spec(
      data = segmentation_data(tempdir(), tempdir(), colormap = binary_colormap),
      models = list(u_net("u", input_shape = c(64, 64),
                          augmentation = list(augment("horizontalflip"))))
    ),
    error = function(e) e
  )
  expect_match(conditionMessage(error), "Did you mean 'HorizontalFlip'")
})

test_that("the problems are available as data, not only as prose", {
  skip_if_no_engine()
  error <- tryCatch(
    platypus_spec(
      data = segmentation_data(tempdir(), tempdir(), colormap = binary_colormap),
      models = list(u_net("u", input_shape = c(64, 64), dropout = 2))
    ),
    error = function(e) e
  )
  expect_s3_class(error$problems, "data.frame")
  expect_true("where" %in% names(error$problems))
  expect_match(error$problems$where[[1]], "models\\[\\[1\\]\\]")
})

test_that("no traceback and no mention of Python reaches the user", {
  # The audience knows their data and their question, and neither knows nor cares that
  # any of this is Python.
  skip_if_no_engine()
  error <- tryCatch(
    platypus_spec(
      data = segmentation_data(tempdir(), tempdir(), colormap = binary_colormap),
      models = list(u_net("u", input_shape = c(64, 64), dropout = 2))
    ),
    error = function(e) e
  )
  message <- conditionMessage(error)
  expect_false(grepl("Traceback|pyplatypus\\.errors|py_last_error", message))
})

test_that("a specification with no models says so plainly", {
  expect_error(
    platypus_spec(data = segmentation_data("a", "b", colormap = binary_colormap),
                  models = list()),
    "at least one model"
  )
})
