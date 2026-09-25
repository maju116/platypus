test_that("a fit needs a specification, and says so", {
  expect_error(platypus_fit("not a spec"), "platypus_spec")
})

test_that("worker count is chosen rather than left at the slow default", {
  # Zero was the obvious default and the wrong one: decoding images in the main process
  # leaves the GPU idle for most of an epoch. Measured on the Data Science Bowl, zero
  # workers took 35 seconds an epoch where eight took under ten.
  expect_type(platypus:::resolve_workers("auto"), "integer")
  expect_lte(platypus:::resolve_workers("auto"), 4L)
  expect_identical(platypus:::resolve_workers(0), 0L)
  expect_identical(platypus:::resolve_workers(7), 7L)
  expect_error(platypus:::resolve_workers("many"), "number")
})

test_that("training returns a fit that knows what it did", {
  skip_if_no_engine()
  fit <- platypus_fit(tiny_spec(tiny_dataset()))
  expect_s3_class(fit, "platypus_fit")
  expect_identical(fit$models, "tiny")
  expect_s3_class(training_history(fit), "data.frame")
  expect_identical(nrow(training_history(fit)), 1L)
})

test_that("the history carries losses, metrics and timing", {
  skip_if_no_engine()
  history <- training_history(platypus_fit(tiny_spec(tiny_dataset())))
  expect_true(all(c("model", "epoch", "train_loss", "val_loss", "val_dice", "seconds")
                  %in% names(history)))
})

test_that("training reports the device it actually used", {
  # A run that quietly fell back to the processor looks exactly like a run that is simply
  # slow, and the difference can be an hour.
  skip_if_no_engine()
  fit <- platypus_fit(tiny_spec(tiny_dataset()))
  expect_true(nzchar(fit$device$device))
  expect_true(nzchar(fit$device$torch))
})

test_that("the comparison table names the loss beside the loss column", {
  # Two models trained on different objectives are not on a common scale, and a bare
  # `loss` column invites exactly that comparison.
  skip_if_no_engine()
  fit <- platypus_fit(tiny_spec(tiny_dataset()))
  table <- evaluate(fit)
  expect_s3_class(table, "data.frame")
  expect_true(all(c("model", "architecture", "loss_function", "parameters", "dice")
                  %in% names(table)))
  expect_identical(table$model, "tiny")
})

test_that("predictions come back as R arrays of class indices", {
  skip_if_no_engine()
  root <- tiny_dataset()
  fit <- platypus_fit(tiny_spec(root))
  masks <- predict(fit, split = "validation")
  expect_true(is.array(masks))
  expect_identical(dim(masks), c(6L, 32L, 32L))
  # One-based, to line up with the colormap. A mask whose background is 0 while its
  # colormap starts at 1 is a trap laid for later.
  expect_true(all(masks %in% c(1L, 2L)))
})

test_that("probabilities are available for anyone who needs them", {
  skip_if_no_engine()
  fit <- platypus_fit(tiny_spec(tiny_dataset()))
  probabilities <- predict(fit, split = "validation", type = "probability")
  expect_identical(dim(probabilities), c(6L, 32L, 32L, 2L))
  expect_true(all(abs(apply(probabilities, 1:3, sum) - 1) < 1e-4))
})

test_that("asking for a model that was not trained lists the ones that were", {
  skip_if_no_engine()
  fit <- platypus_fit(tiny_spec(tiny_dataset()))
  expect_error(predict(fit, "nonexistent"), "tiny")
})

test_that("tiling returns masks at the source size, reassembled", {
  # The capability the package of 2020 lacked: an image cut into a grid came back as
  # pieces. Here six tiles of 32 become one 64x96 mask.
  skip_if_no_engine()
  root <- tiny_dataset(n = 3, size = 96)
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("tiled", input_shape = c(32, 32), blocks = 2, filters = 4,
                        splits = c(2, 3), epochs = 1, batch_size = 2))
  )
  masks <- predict(platypus_fit(spec), split = "validation")
  expect_identical(dim(masks), c(3L, 64L, 96L))
})

test_that("training a model that cannot learn still returns a usable fit", {
  # Failure in the middle of a long run is the most expensive moment to stop being
  # readable, so the paths that report it are worth exercising.
  skip_if_no_engine()
  fit <- platypus_fit(tiny_spec(tiny_dataset(), loss = loss_lovasz()))
  expect_s3_class(fit, "platypus_fit")
  expect_true(is.finite(training_history(fit)$train_loss[[1]]))
})
