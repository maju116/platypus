context("segmentation_generator")

test_that("segmentation_generator accepts only correct inputs.", {
  path <- ""
  colormap <- list(c(0, 0, 0), c(111, 111, 111), c(255, 255, 255))
  correct_mode <- "nested_dirs"
  correct_only_images <- TRUE
  correct_net_h <- 256
  correct_net_w <- 256
  correct_grayscale <- TRUE
  scale <- 1 / 255
  batch_size <- 1
  correct_shuffle <- FALSE
  subdirs <- c("images", "masks")
  column_sep <- ";"

  incorrect_mode_1 <- "config"
  incorrect_mode_2 <- 4
  incorrect_only_images <- NA
  incorrect_net_h <- 255
  incorrect_net_w <- 111
  incorrect_grayscale <- NA
  incorrect_shuffle <- NA

  expect_output(segmentation_generator(path, colormap, correct_mode, correct_only_images, correct_net_h, correct_net_w,
                                       correct_grayscale, scale, batch_size, correct_shuffle,
                                       subdirs, column_sep))
  expect_error(segmentation_generator(path, colormap, incorrect_mode_1, correct_only_images, correct_net_h, correct_net_w,
                                      correct_grayscale, scale, batch_size, correct_shuffle,
                                      subdirs, column_sep))
  expect_error(segmentation_generator(path, colormap, incorrect_mode_2, correct_only_images, correct_net_h, correct_net_w,
                                      correct_grayscale, scale, batch_size, correct_shuffle,
                                      subdirs, column_sep))
  expect_error(segmentation_generator(path, colormap, correct_mode, incorrect_only_images, correct_net_h, correct_net_w,
                                      correct_grayscale, scale, batch_size, correct_shuffle,
                                      subdirs, column_sep))
  expect_error(segmentation_generator(path, colormap, correct_mode, correct_only_images, incorrect_net_h, correct_net_w,
                                      correct_grayscale, scale, batch_size, correct_shuffle,
                                      subdirs, column_sep))
  expect_error(segmentation_generator(path, colormap, correct_mode, correct_only_images, correct_net_h, incorrect_net_w,
                                      correct_grayscale, scale, batch_size, correct_shuffle,
                                      subdirs, column_sep))
  expect_error(segmentation_generator(path, colormap, correct_mode, correct_only_images, correct_net_h, correct_net_w,
                                      incorrect_grayscale, scale, batch_size, correct_shuffle,
                                      subdirs, column_sep))
  expect_error(segmentation_generator(path, colormap, correct_mode, correct_only_images, correct_net_h, correct_net_w,
                                      correct_grayscale, scale, batch_size, incorrect_shuffle,
                                      subdirs, column_sep))
})

test_that("segmentation_generator reads images and masks correctly from directory", {
  testdata_path <- system.file("testdata", package = "platypus")
  path <- file.path(testdata_path, "dir")
  mode <- "dir"
  colormap <- list(c(0, 0, 0), c(111, 111, 111),
                   c(222, 222, 222), c(255, 255, 255))
  only_images <- FALSE
  net_h <- 2
  net_w <- 2
  grayscale <- FALSE
  scale <- 1 / 255
  batch_size <- 3
  shuffle <- FALSE
  subdirs <- c("images", "masks")

  sample_generator <- segmentation_generator(path, colormap, mode, only_images, net_h, net_w,
                                             grayscale, scale, batch_size, shuffle, subdirs)
  generator_output <- keras::generator_next(sample_generator)
  expected_output <- list(
    list(
      array(c(255, 111, 0, 222,
              255, 111, 0, 222,
              255, 111, 0, 222), dim = c(2, 2, 3)),
      array(c(255, 111, 222, 0,
              255, 111, 222, 0,
              255, 111, 222, 0), dim = c(2, 2, 3)),
      array(c(222, 111, 0, 255,
              222, 111, 0, 255,
              222, 111, 0, 255), dim = c(2, 2, 3))
    ) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3)) %>% `/`(255),
    list(
      list(
        array(c(0, 0, 1, 0,
                0, 0, 1, 0,
                0, 0, 1, 0), dim = c(2, 2)),
        array(c(0, 1, 0, 0,
                0, 1, 0, 0,
                0, 1, 0, 0), dim = c(2, 2)),
        array(c(0, 0, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1), dim = c(2, 2)),
        array(c(1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0), dim = c(2, 2))
      ) %>% abind::abind(along = 3),
      list(
        array(c(0, 0, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1), dim = c(2, 2)),
        array(c(0, 1, 0, 0,
                0, 1, 0, 0,
                0, 1, 0, 0), dim = c(2, 2)),
        array(c(0, 0, 1, 0,
                0, 0, 1, 0,
                0, 0, 1, 0), dim = c(2, 2)),
        array(c(1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0), dim = c(2, 2))
      ) %>% abind::abind(along = 3),
      list(
        array(c(0, 0, 1, 0,
                0, 0, 1, 0,
                0, 0, 1, 0), dim = c(2, 2)),
        array(c(0, 1, 0, 0,
                0, 1, 0, 0,
                0, 1, 0, 0), dim = c(2, 2)),
        array(c(1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0), dim = c(2, 2)),
        array(c(0, 0, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1), dim = c(2, 2))
      ) %>% abind::abind(along = 3)
    ) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3))
  )

  expect_true(all.equal(expected_output[[1]], generator_output[[1]], check.attributes = FALSE))
  expect_true(all.equal(expected_output[[2]], generator_output[[2]], check.attributes = FALSE))
})

test_that("segmentation_generator reads images and masks correctly from nested directories", {
  testdata_path <- system.file("testdata", package = "platypus")
  path <- file.path(testdata_path, "nested_dirs")
  mode <- "nested_dirs"
  colormap <- list(c(0, 0, 0), c(111, 111, 111),
                   c(222, 222, 222), c(255, 255, 255))
  only_images <- FALSE
  net_h <- 2
  net_w <- 2
  grayscale <- FALSE
  scale <- 1 / 255
  batch_size <- 3
  shuffle <- FALSE
  subdirs <- c("images", "masks")

  sample_generator <- segmentation_generator(path, colormap, mode, only_images, net_h, net_w,
                                             grayscale, scale, batch_size, shuffle, subdirs)
  generator_output <- keras::generator_next(sample_generator)
  expected_output <- list(
    list(
      array(c(255, 111, 0, 222,
              255, 111, 0, 222,
              255, 111, 0, 222), dim = c(2, 2, 3)),
      array(c(255, 111, 222, 0,
              255, 111, 222, 0,
              255, 111, 222, 0), dim = c(2, 2, 3)),
      array(c(222, 111, 0, 255,
              222, 111, 0, 255,
              222, 111, 0, 255), dim = c(2, 2, 3))
    ) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3)) %>% `/`(255),
    list(
      list(
        array(c(0, 0, 1, 0,
                0, 0, 1, 0,
                0, 0, 1, 0), dim = c(2, 2)),
        array(c(0, 1, 0, 0,
                0, 1, 0, 0,
                0, 1, 0, 0), dim = c(2, 2)),
        array(c(0, 0, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1), dim = c(2, 2)),
        array(c(1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0), dim = c(2, 2))
      ) %>% abind::abind(along = 3),
      list(
        array(c(0, 0, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1), dim = c(2, 2)),
        array(c(0, 1, 0, 0,
                0, 1, 0, 0,
                0, 1, 0, 0), dim = c(2, 2)),
        array(c(0, 0, 1, 0,
                0, 0, 1, 0,
                0, 0, 1, 0), dim = c(2, 2)),
        array(c(1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0), dim = c(2, 2))
      ) %>% abind::abind(along = 3),
      list(
        array(c(0, 0, 1, 0,
                0, 0, 1, 0,
                0, 0, 1, 0), dim = c(2, 2)),
        array(c(0, 1, 0, 0,
                0, 1, 0, 0,
                0, 1, 0, 0), dim = c(2, 2)),
        array(c(1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0), dim = c(2, 2)),
        array(c(0, 0, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1), dim = c(2, 2))
      ) %>% abind::abind(along = 3)
    ) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3))
  )

  expect_true(all.equal(expected_output[[1]], generator_output[[1]], check.attributes = FALSE))
  expect_true(all.equal(expected_output[[2]], generator_output[[2]], check.attributes = FALSE))
})

test_that("segmentation_generator reads images and masks correctly from configuration CSV", {
  testdata_path <- system.file("testdata", package = "platypus")
  path <- file.path(tempdir(), "sample_config_file.csv")
  nested_dirs <- list.dirs(file.path(testdata_path, "nested_dirs"), full.names  = TRUE, recursive = FALSE)
  images_paths <- nested_dirs %>% purrr::map(~ list.files(file.path(.x, "images"), full.names  = TRUE))
  masks_paths <- nested_dirs %>% purrr::map(~ paste0(list.files(file.path(.x, "masks"), full.names  = TRUE), collapse = ";"))
  config_df <- data.frame(images = unlist(images_paths), masks = unlist(masks_paths))
  readr::write_csv(config_df, path = path)

  mode <- "config_file"
  colormap <- list(c(0, 0, 0), c(111, 111, 111),
                   c(222, 222, 222), c(255, 255, 255))
  only_images <- FALSE
  net_h <- 2
  net_w <- 2
  grayscale <- FALSE
  scale <- 1 / 255
  batch_size <- 3
  shuffle <- FALSE
  subdirs <- c("images", "masks")

  sample_generator <- segmentation_generator(path, colormap, mode, only_images, net_h, net_w,
                                             grayscale, scale, batch_size, shuffle, subdirs)
  generator_output <- keras::generator_next(sample_generator)
  expected_output <- list(
    list(
      array(c(255, 111, 0, 222,
              255, 111, 0, 222,
              255, 111, 0, 222), dim = c(2, 2, 3)),
      array(c(255, 111, 222, 0,
              255, 111, 222, 0,
              255, 111, 222, 0), dim = c(2, 2, 3)),
      array(c(222, 111, 0, 255,
              222, 111, 0, 255,
              222, 111, 0, 255), dim = c(2, 2, 3))
    ) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3)) %>% `/`(255),
    list(
      list(
        array(c(0, 0, 1, 0,
                0, 0, 1, 0,
                0, 0, 1, 0), dim = c(2, 2)),
        array(c(0, 1, 0, 0,
                0, 1, 0, 0,
                0, 1, 0, 0), dim = c(2, 2)),
        array(c(0, 0, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1), dim = c(2, 2)),
        array(c(1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0), dim = c(2, 2))
      ) %>% abind::abind(along = 3),
      list(
        array(c(0, 0, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1), dim = c(2, 2)),
        array(c(0, 1, 0, 0,
                0, 1, 0, 0,
                0, 1, 0, 0), dim = c(2, 2)),
        array(c(0, 0, 1, 0,
                0, 0, 1, 0,
                0, 0, 1, 0), dim = c(2, 2)),
        array(c(1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0), dim = c(2, 2))
      ) %>% abind::abind(along = 3),
      list(
        array(c(0, 0, 1, 0,
                0, 0, 1, 0,
                0, 0, 1, 0), dim = c(2, 2)),
        array(c(0, 1, 0, 0,
                0, 1, 0, 0,
                0, 1, 0, 0), dim = c(2, 2)),
        array(c(1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0), dim = c(2, 2)),
        array(c(0, 0, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1), dim = c(2, 2))
      ) %>% abind::abind(along = 3)
    ) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3))
  )

  expect_true(all.equal(expected_output[[1]], generator_output[[1]], check.attributes = FALSE))
  expect_true(all.equal(expected_output[[2]], generator_output[[2]], check.attributes = FALSE))


})
