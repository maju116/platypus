# Weights by name, and saving your own. Nothing here downloads anything: fetching from the Hub is
# the engine's job and is tested there, offline. What belongs here is the surface - the listing
# arriving as a data frame, and a trained model's weights making the round trip through R.

test_that("the published weights are listed with the commit each name is pinned to", {
  skip_if_no_weights_registry()
  listing <- available_weights()

  expect_s3_class(listing, "data.frame")
  expect_true(all(c("name", "repo", "filename", "revision", "description") %in% names(listing)))
  expect_true(nrow(listing) >= 1)
  # A full commit hash, not a branch: a branch can be moved to different numbers later.
  expect_true(all(grepl("^[0-9a-f]{40}$", listing$revision)))
  expect_true(all(grepl("[.]safetensors$", listing$filename)))
})

test_that("the nuclei entry says what it cannot do", {
  # Someone chooses weights by reading this. The limitation decides whether they answer the right
  # question, so it has to be in the description and not only in a paper somewhere.
  skip_if_no_weights_registry()
  listing <- available_weights()
  nuclei <- listing[listing$name == "dsbowl-unet", ]

  expect_equal(nrow(nuclei), 1)
  expect_match(nuclei$description, "BBBC038")
  expect_match(nuclei$description, "[Ss]emantic")
})

test_that("a trained model's weights make the round trip through R", {
  skip_if_no_weights_registry()
  root <- tiny_dataset()
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("tiny", input_shape = c(32, 32), blocks = 2, filters = 4,
                        epochs = 1, batch_size = 2,
                        metrics = list(metric_dice())))
  )
  fit <- platypus_fit(spec, num_workers = 0)

  out <- file.path(withr::local_tempdir(), "mine")
  written <- export_weights(fit, out, data = "the tiny fixture", licence = "MIT")
  expect_true(file.exists(written))
  expect_true(file.exists(sub("[.]safetensors$", ".json", written)))

  # The sidecar comes back attached, so seeing what was recorded needs no JSON reader in R.
  recorded <- attr(written, "recorded")
  expect_equal(unlist(recorded$input_shape), c(32L, 32L))
  # Whatever was passed rides along: provenance belongs with the file.
  expect_equal(recorded$data, "the tiny fixture")
  expect_equal(recorded$licence, "MIT")

  # And they load back into the same specification, which is the point of saving them.
  again <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("tiny", input_shape = c(32, 32), blocks = 2, filters = 4,
                        epochs = 1, batch_size = 2, weights = written, fit = FALSE,
                        metrics = list(metric_dice())))
  )
  loaded <- platypus_fit(again, num_workers = 0)
  expect_equal(dim(predict(loaded, split = "validation")), c(6L, 32L, 32L))
})

test_that("weights that belong to another model are refused with the field that differs", {
  # The sidecar's whole purpose. A wrong channel count fails inside torch anyway; a different
  # colormap with the same class count would load cleanly and predict nonsense.
  skip_if_no_weights_registry()
  root <- tiny_dataset()
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("tiny", input_shape = c(32, 32), blocks = 2, filters = 4,
                        epochs = 1, batch_size = 2))
  )
  written <- export_weights(platypus_fit(spec, num_workers = 0),
                            file.path(withr::local_tempdir(), "mine"))

  wider <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("tiny", input_shape = c(32, 32), blocks = 2, filters = 8,
                        epochs = 1, batch_size = 2, weights = written, fit = FALSE))
  )
  expect_error(platypus_fit(wider, num_workers = 0), "filters")
})

test_that("an unknown name lists the published ones", {
  skip_if_no_weights_registry()
  root <- tiny_dataset()
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("tiny", input_shape = c(32, 32), blocks = 2, filters = 4,
                        epochs = 1, batch_size = 2, weights = "not-a-thing", fit = FALSE))
  )
  expect_error(platypus_fit(spec, num_workers = 0), "no published weights")
})

test_that("export_weights refuses what it cannot save", {
  expect_error(export_weights(list(), "somewhere"), "platypus_fit")
})
