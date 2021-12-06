context("u_net TRAINING RUNS")

test_that("u_net model can be fitted with no errors", {
  blocks <- 3
  classes <- 4
  filters <- 12
  dropout <- 0.567
  batch_normalization <- TRUE
  testdata_path <- system.file("testdata", package = "platypus")
  path <- file.path(testdata_path, "dir")
  mode <- "dir"
  colormap <- list(c(0, 0, 0), c(111, 111, 111),
                   c(222, 222, 222), c(255, 255, 255))
  only_images <- FALSE
  net_h <- 64
  net_w <- 64
  grayscale <- FALSE
  scale <- 1 / 255
  batch_size <- 3
  shuffle <- FALSE
  subdirs <- c("images", "masks")
  model <- u_net(net_h, net_w, grayscale, blocks, classes, filters,
                  dropout, batch_normalization)
  model %>% compile(
    optimizer = optimizer_adam(lr = 1e-3),
    loss = loss_dice(),
    metrics = metric_dice_coeff()
  )
  model
  sample_generator <- segmentation_generator(path, colormap, mode, only_images, net_h, net_w,
                                             grayscale, scale, batch_size, shuffle, subdirs)
  start_weights <- get_weights(model)
  model %>% fit_generator(sample_generator, steps_per_epoch = 1, epochs = 2)
  end_weights <- get_weights(model)
  expect_true(!identical(start_weights, end_weights))
})
