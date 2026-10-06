test_that("a specification built from arguments reaches the engine intact", {
  skip_if_no_engine()
  spec <- platypus_spec(
    data = segmentation_data(tempdir(), tempdir(), colormap = binary_colormap),
    models = list(u_net("unet", input_shape = c(256, 256), blocks = 4, filters = 16))
  )
  expect_s3_class(spec, "platypus_spec")

  built <- as.list(spec)
  expect_length(built$models, 1L)
  expect_identical(built$models[[1]]$name, "unet")
  expect_identical(built$models[[1]]$architecture, "u_net")
  expect_equal(unlist(built$models[[1]]$input_shape), c(256L, 256L))
})

test_that("the engine's defaults apply to anything left unset", {
  # R keeps NULLs in a list, so an unset argument would otherwise cross over as an
  # explicit None and overwrite the default on the other side.
  skip_if_no_engine()
  built <- as.list(platypus_spec(
    data = segmentation_data(tempdir(), tempdir(), colormap = binary_colormap),
    models = list(u_net("unet", input_shape = c(64, 64)))
  ))
  model <- built$models[[1]]
  expect_identical(model$loss$name, "cce")
  expect_identical(model$optimizer$name, "adam")
  expect_equal(model$blocks, 4L)
  expect_null(model$splits)
})

test_that("arguments and YAML produce the same specification", {
  # The contract the whole design rests on: a colleague's configuration file and your R
  # script describe the same experiment, and nothing downstream can tell them apart.
  skip_if_no_engine()
  path <- withr::local_tempfile(fileext = ".yaml")
  writeLines(
    gsub("PLACEHOLDER", tempdir(),
         readLines(test_path("fixtures", "experiment.yaml")), fixed = TRUE),
    path
  )

  from_file <- platypus_spec(path)
  from_code <- platypus_spec(
    data = segmentation_data(tempdir(), tempdir(), colormap = binary_colormap),
    models = list(u_net(
      "unet", input_shape = c(256, 256), channels = 3,
      blocks = 4, filters = 16,
      loss = loss_cce_dice(cce_weight = 0.5),
      metrics = list(metric_dice(include_background = FALSE)),
      optimizer = optimizer_adam(learning_rate = 0.001),
      epochs = 20, batch_size = 8
    ))
  )

  expect_identical(as.list(from_file), as.list(from_code))
})

test_that("numbers arrive as the integers the engine expects", {
  # R numerics are doubles, so a plain 4 crosses as 4.0. Leaving that to coercion works
  # until it does not, and then the message comes from a library the user never invoked.
  skip_if_no_engine()
  built <- as.list(platypus_spec(
    data = segmentation_data(tempdir(), tempdir(), colormap = binary_colormap),
    models = list(u_net("unet", input_shape = c(64, 64), blocks = 3, filters = 8,
                        epochs = 5, batch_size = 4, splits = c(2, 2)))
  ))
  model <- built$models[[1]]
  expect_identical(unlist(model$input_shape), c(64L, 64L))
  expect_identical(unlist(model$splits), c(2L, 2L))
  expect_equal(model$epochs, 5L)
})

test_that("several models live in one specification", {
  skip_if_no_engine()
  spec <- platypus_spec(
    data = segmentation_data(tempdir(), tempdir(), colormap = binary_colormap),
    models = list(
      u_net("a", input_shape = c(64, 64)),
      linknet("b", input_shape = c(64, 64), loss = loss_focal_tversky(alpha = 0.7)),
      res_u_net("c", input_shape = c(64, 64)),
      u_net_plus_plus("d", input_shape = c(64, 64), deep_supervision = TRUE)
    )
  )
  built <- as.list(spec)
  expect_identical(vapply(built$models, `[[`, character(1), "architecture"),
                   c("u_net", "linknet", "res_u_net", "u_net_plus_plus"))
})

test_that("a 3D specification is accepted", {
  # Rank comes from the length of input_shape, not a switch. The data pipeline stops at
  # 2D for now, but nothing in the specification does.
  skip_if_no_engine()
  built <- as.list(platypus_spec(
    data = segmentation_data(tempdir(), tempdir(), colormap = binary_colormap),
    models = list(u_net("volume", input_shape = c(32, 32, 32), blocks = 2,
                        splits = c(2, 2, 2)))
  ))
  expect_length(built$models[[1]]$input_shape, 3L)
})

# --- dividing one folder instead of naming three ------------------------------------------

test_that("split and validation are exactly one of each", {
  expect_error(segmentation_data("t", colormap = binary_colormap),
               "something to validate against")
  expect_error(
    segmentation_data("t", "v", colormap = binary_colormap,
                      split = list(fractions = c(0.8, 0.2), group_by = NULL)),
    "something to validate against"
  )
})

test_that("split must answer group_by, even with NULL", {
  # The field the whole block exists to put in front of someone: slices of one patient on
  # both sides make validation measure memory, and nothing in the output says so.
  expect_error(
    segmentation_data("t", colormap = binary_colormap,
                      split = list(fractions = c(0.8, 0.2))),
    "must say `group_by`"
  )
})

test_that("a NULL group_by survives the crossing into the engine", {
  # The bug neither suite could have caught alone. `compact()` drops a NULL because
  # everywhere else it means "not asked for"; here it is a stated answer, and the engine
  # requires the key so that nobody leaves the question open. Both rules are right and
  # they met here.
  skip_if_no_split_block()
  data <- segmentation_data("t", colormap = binary_colormap,
                            split = list(fractions = c(0.8, 0.2), group_by = NULL))
  expect_true("group_by" %in% names(data$split))
  expect_null(data$split$group_by)
  expect_silent(invisible(platypus_spec(
    data = data, models = list(u_net("u", input_shape = c(32, 32))), check_paths = FALSE
  )))
})

test_that("test and split do not both claim the test set", {
  expect_error(
    segmentation_data("t", colormap = binary_colormap, test = "x",
                      split = list(fractions = c(0.6, 0.2, 0.2), group_by = NULL)),
    "both say where the test set comes from"
  )
})

# --- loss_boundary -----------------------------------------------------------------------

test_that("loss_boundary sends nothing it was not given", {
  # The default alpha is a measured number - 0.9, because 0.5 was measured and is unusable -
  # so it lives in the engine and R must not restate it. Restating would mean an engine that
  # learns better being silently contradicted from this side.
  loss <- loss_boundary()
  expect_identical(loss$name, "boundary")
  expect_null(loss$region)
  expect_null(loss$alpha)
})

test_that("an alpha outside the open interval is refused with the reason", {
  for (bad in list(0, 1, -0.2, 1.5)) {
    expect_error(loss_boundary(alpha = bad), "strictly between 0 and 1")
  }
  expect_error(loss_boundary(alpha = "half"), "one number")
  expect_error(loss_boundary(alpha = c(0.5, 0.6)), "one number")
})

test_that("a boundary loss cannot be its own region term", {
  expect_error(loss_boundary(region = loss_boundary()), "cannot work")
})

test_that("any region loss can be combined with it, which is what the issue asked for", {
  loss <- loss_boundary(region = loss_focal(gamma = 3))
  expect_identical(loss$region$name, "focal")
  expect_identical(loss$region$gamma, 3)
})

test_that("the engine accepts it, nested region and all", {
  skip_if_no_boundary_loss()
  spec <- platypus_spec(
    data = segmentation_data("a", "b", colormap = binary_colormap),
    models = list(u_net("u", input_shape = c(32, 32),
                        loss = loss_boundary(region = loss_focal(gamma = 3),
                                             alpha = 0.8))),
    check_paths = FALSE
  )
  built <- as.list(spec)$models[[1]]$loss
  expect_identical(built$name, "boundary")
  expect_identical(built$region$name, "focal")
  expect_equal(built$alpha, 0.8)
})

test_that("unset, the engine's measured default arrives rather than one of ours", {
  skip_if_no_boundary_loss()
  spec <- platypus_spec(
    data = segmentation_data("a", "b", colormap = binary_colormap),
    models = list(u_net("u", input_shape = c(32, 32), loss = loss_boundary())),
    check_paths = FALSE
  )
  built <- as.list(spec)$models[[1]]$loss
  expect_equal(built$alpha, 0.9)          # the measured value, from the engine
  expect_identical(built$region$name, "dice")
})

# --- callback_swa ------------------------------------------------------------------------

test_that("callback_swa sends nothing it was not given", {
  # `start = 0.75` is a measured value - the arm at 0.5 was worse on every column - so it
  # lives in the engine and R must not restate it.
  cb <- callback_swa()
  expect_identical(cb$name, "swa")
  expect_null(cb$start)
  expect_null(cb$learning_rate)
})

test_that("a start outside (0, 1] is refused, and so is a non-positive rate", {
  for (bad in list(0, -0.1, 1.5, "half", c(0.5, 0.6))) {
    expect_error(callback_swa(start = bad), "above 0 and at most 1")
  }
  for (bad in list(0, -1, "fast")) {
    expect_error(callback_swa(learning_rate = bad), "one positive number")
  }
})

test_that("the engine takes it, and refuses it beside a cosine", {
  skip_if_no_swa()
  data <- segmentation_data("a", "b", colormap = binary_colormap)
  spec <- platypus_spec(
    data = data,
    models = list(u_net("u", input_shape = c(32, 32),
                        callbacks = list(callback_swa(start = 0.6,
                                                      learning_rate = 1e-3)))),
    check_paths = FALSE
  )
  built <- as.list(spec)$models[[1]]$callbacks[[1]]
  expect_identical(built$name, "swa")
  expect_equal(built$start, 0.6)

  # Both configured, neither complaining, and the average would be the last epoch: the one
  # combination worth refusing rather than documenting.
  expect_error(
    platypus_spec(
      data = data,
      models = list(u_net("u", input_shape = c(32, 32),
                          callbacks = list(callback_swa(),
                                           callback_cosine_annealing()))),
      check_paths = FALSE
    ),
    "undo each other"
  )
})

test_that("unset, the engine's measured default arrives", {
  skip_if_no_swa()
  spec <- platypus_spec(
    data = segmentation_data("a", "b", colormap = binary_colormap),
    models = list(u_net("u", input_shape = c(32, 32),
                        callbacks = list(callback_swa()))),
    check_paths = FALSE
  )
  built <- as.list(spec)$models[[1]]$callbacks[[1]]
  expect_equal(built$start, 0.75)
  expect_null(built$learning_rate)
})
