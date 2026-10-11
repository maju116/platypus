# `read_record()`, and what it is held to.
#
# The engine wrote records from the beginning and this package had no way to read one, which
# is how `tests/testthat/test-detection.R` came to call `jsonlite::fromJSON()` directly: the
# suite needed to check that a detector's fitted anchors had been written down, and there was
# no function to do it with. A capability in the tests and not in the package.
#
# What is asserted below is deliberately split. The cheap half needs no engine and covers the
# contract a caller meets when something is missing. The expensive half trains a model for one
# epoch, because the only honest way to test a reader is against a file the writer wrote -
# this file's idea of a record is exactly the thing that would drift.

test_that("a path that holds no record says so in terms of what the caller did", {
  empty <- withr::local_tempdir()

  # Not jsonlite's "cannot open connection": the ordinary way to arrive here is a run whose
  # specification never named `output_dir`, in which case nothing was written and nothing is
  # missing. The message has to name that, or the reader goes looking for a corrupt file.
  expect_error(read_record(empty), "No run record at")
  expect_error(read_record(empty), "named an `output_dir`", fixed = TRUE)
  expect_error(read_record(file.path(empty, "run.json")), "No run record at")
})

test_that("`path` is one name, and anything else is refused before any file is touched", {
  expect_error(read_record(c("a", "b")), "one file or directory", fixed = TRUE)
  expect_error(read_record(NA_character_), "one file or directory", fixed = TRUE)
  expect_error(read_record(42), "one file or directory", fixed = TRUE)
})

test_that("a record written by a real run reads back whole", {
  skip_if_no_engine()

  root <- tiny_dataset()
  out <- withr::local_tempdir()
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("tiny", input_shape = c(32, 32), blocks = 2, filters = 4,
                        epochs = 1, batch_size = 2)),
    output_dir = out
  )
  fit <- platypus_fit(spec, device = "cpu")

  # Both forms, because the directory is the one a caller has to hand and the file is the
  # one the engine's documentation names.
  by_directory <- read_record(file.path(out, "tiny"))
  by_file <- read_record(file.path(out, "tiny", "run.json"))
  expect_identical(by_directory, by_file)

  record <- by_directory
  expect_s3_class(record, "platypus_record")

  # Held to the writer rather than to this file's idea of a record: these are the keys
  # `pyplatypus/runs.py` writes, and a release that stopped writing one would fail here
  # rather than returning NULL somewhere later.
  expect_setequal(
    names(record),
    c("written", "pyplatypus", "model", "specification", "derived", "history")
  )
  expect_identical(record$model, "tiny")

  # The engine stamps its own version into every record, which is the provenance that
  # outlives the run - and the pin is what it should be.
  expect_identical(record$pyplatypus, platypus_status()$engine_version)

  # The one thing R does to the record rather than handing it over: history as a frame,
  # the same shape `training_history()` gives for a fit in this session.
  expect_s3_class(record$history, "data.frame")
  expect_identical(nrow(record$history), 1L)
  expect_true("epoch" %in% names(record$history))

  # The one difference between the two frames, asserted as the difference rather than
  # assumed away. A record belongs to one model and names it at the top level, so there is
  # no `model` column; `training_history()` spans the fit's models and needs one. The first
  # version of this test asserted `model` was in both and was right to fail - the
  # documentation had claimed "the same shape", which is the kind of nearly-true that gets
  # somebody's `rbind()` later.
  live <- training_history(fit)
  expect_identical(setdiff(names(live), names(record$history)), "model")
  expect_identical(setdiff(names(record$history), names(live)), character(0))
  expect_identical(nrow(record$history), nrow(live))

  # A segmentation run derives nothing, and the engine's own docstring calls that the honest
  # answer rather than an omission: the specification already carries the colormap, the
  # shape and the window, so re-running it reproduces the model.
  expect_true(is.list(record$derived))
  expect_length(record$derived, 0L)

  # "A thing you can run again": the specification is in there whole, not summarised.
  expect_true(all(c("task", "data", "models") %in% names(record$specification)))
  expect_identical(record$specification$models[[1]]$name, "tiny")
})

test_that("printing a record says what it is without dumping the specification", {
  skip_if_no_engine()

  root <- tiny_dataset()
  out <- withr::local_tempdir()
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("printed", input_shape = c(32, 32), blocks = 2, filters = 4,
                        epochs = 1, batch_size = 2)),
    output_dir = out
  )
  platypus_fit(spec, device = "cpu")

  shown <- paste(capture.output(print(read_record(file.path(out, "printed")))),
                 collapse = "\n")

  expect_match(shown, "printed", fixed = TRUE)
  expect_match(shown, "pyplatypus", fixed = TRUE)
  expect_match(shown, "epochs", fixed = TRUE)
  # `derived` is named rather than counted, because for a detector the name that matters is
  # `anchors` and "1 derived" would not say it.
  expect_match(shown, "derived", fixed = TRUE)
  # And the specification is summarised, not printed: a record carries the whole thing, and
  # a print method that dumps it is a print method nobody calls twice.
  expect_false(grepl("colormap", shown, fixed = TRUE))
})
