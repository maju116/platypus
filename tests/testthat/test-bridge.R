test_that("loading the package does not start Python", {
  # The reason library(platypus) is instant and works on a machine that has never seen
  # Python. If this ever fails, delay_load has been lost somewhere.
  #
  # In a fresh process on purpose. Asking the current session would make the answer
  # depend on whether anything else had already started the engine, so the test would
  # quietly stop meaning anything the moment another test ran first.
  skip_if_not_installed("callr")
  started <- callr::r(function() {
    library(platypus)
    reticulate::py_available(initialize = FALSE)
  })
  expect_false(started)
})

test_that("status reports the pinned engine before anything has started", {
  status <- platypus_status()
  expect_s3_class(status, "platypus_status")
  # The pinned version is what matters; the extras depend on whether
  # platypus_use_torch() has been called, which is none of this test's business.
  # PLATYPUS_ENGINE_PATH replaces the requirement with a source tree during development.
  if (nzchar(Sys.getenv("PLATYPUS_ENGINE_PATH"))) {
    expect_true(dir.exists(status$requirement))
  } else {
    expect_match(status$requirement, "^pyplatypus(\\[[a-z]+\\])?==0\\.2\\.0a1$")
  }
  expect_type(status$started, "logical")
})

test_that("status prints something a person can read", {
  output <- capture.output(print(platypus_status()))
  expect_true(any(grepl("platypus engine", output)))
  expect_true(any(grepl("requires", output)))
})

test_that("a broken RETICULATE_PYTHON is reported rather than left to fail later", {
  # It overrides py_require() entirely, so silence here means an unreadable failure
  # somewhere inside reticulate much later.
  withr::with_envvar(c(RETICULATE_PYTHON = "/definitely/not/a/python"), {
    expect_message(
      platypus:::check_reticulate_python(),
      "does not exist"
    )
  })
})

test_that("a RETICULATE_PYTHON that exists is still called out", {
  withr::with_envvar(c(RETICULATE_PYTHON = tempdir()), {
    expect_message(
      platypus:::check_reticulate_python(),
      "instead of the isolated environment"
    )
  })
})

test_that("an unset RETICULATE_PYTHON says nothing at all", {
  withr::with_envvar(c(RETICULATE_PYTHON = NA), {
    expect_silent(platypus:::check_reticulate_python())
  })
})

test_that("the engine starts and is the version this package pins", {
  skip_if_no_engine()
  status <- platypus_status()
  expect_true(status$started)
  expect_identical(status$engine_version, "0.2.0a1")
  expect_true(nzchar(status$python))
})

test_that("data crosses back from Python as ordinary R objects", {
  # The bridge is only useful if what returns is something R code can work with, rather
  # than a handle a user has to keep asking Python about.
  skip_if_no_engine()
  schema <- platypus:::engine()$spec_schema()
  expect_type(schema, "list")
  expect_true(grepl("spec.schema.json", schema[["$id"]], fixed = TRUE))
})

test_that("a bad specification comes back as a readable R error, not a traceback", {
  # The hardest requirement of the whole surface, and the reason the Python side reports
  # its problems as plain data. Fleshed out properly in the next step; this pins the
  # shape now so it cannot regress silently.
  skip_if_no_engine()
  broken <- list(
    data = list(train_path = tempdir(), validation_path = tempdir(),
                colormap = list(c(0L, 0L, 0L), c(255L, 255L, 255L))),
    models = list(list(name = "u", input_shape = c(64L, 64L),
                       loss = list(name = "focaal")))
  )
  error <- tryCatch(
    platypus:::engine()$from_dict(broken, check_paths = FALSE),
    error = function(e) e
  )
  expect_s3_class(error, "error")
  expect_match(conditionMessage(error), "focaal")
})

test_that("the torch build can be chosen, but only before the engine starts", {
  # A GTX 10-series card needs the CUDA 12 line, because CUDA 13 dropped Pascal. Once
  # Python is running the environment is fixed, and saying so is kinder than silently
  # doing nothing.
  skip_if_no_engine()
  expect_error(platypus_use_torch("pascal"), "already started")
})

test_that("the device report says where the work will happen", {
  skip_if_no_engine()
  device <- platypus_device()
  expect_s3_class(device, "platypus_device")
  expect_true(nzchar(device$torch))
  expect_type(device$cuda_available, "logical")
})
