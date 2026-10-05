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
