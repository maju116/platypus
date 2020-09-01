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

# test_that("segmentation_generator reads images and masks correctly from directory", {
#   testdata_path <- system.file("testdata", package = "platypus")
#   path <- file.path(testdata_path, "dir")
#   mode <- "dir"
#   classes <- 3
#   only_images <- FALSE
#   net_h <- 2
#   net_w <- 2
#   grayscale <- FALSE
#   scale <- 1 / 255
#   batch_size <- 3
#   shuffle <- FALSE
#   subdirs <- c("images", "masks")
#
#   sample_generator <- segmentation_generator(path, colormap, mode, only_images, net_h, net_w,
#                                              grayscale, scale, batch_size, shuffle, subdirs)
#   generator_output <- sample_generator()
#
#   set.seed(666)
#   sample_images <- array(sample(0:255, 3 * 2 * 2 * 3, replace = TRUE), dim = c(3, 2, 2, 3)) / 255
#   sample_masks <- array(sample(c(0, 255), 3 * 2 * 2 * 3, replace = TRUE),
#                         dim = c(3, 2, 2, 3))
#   sample_masks <- 1:3 %>% purrr::map(~ {
#     grayscale <- (0.2980392 * sample_masks[.x, , , 1] + 0.5843137 * sample_masks[.x, , , 2] + 0.1137255 * sample_masks[.x, , , 3])
#     array_reshape(grayscale / 255, c(2, 2, 1))
#   }) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3))
#   sample_masks[sample_masks > 0.99] <- 1 # It somehow makes difference for to_categorical function
#
#   expect_true(all(abs(sample_images - generator_output[[1]]) < 0.005))
#   expect_equal(sample_masks %>% keras::to_categorical(classes), generator_output[[2]])
# })

# test_that("segmentation_generator reads images and masks correctly from nested directories", {
#   testdata_path <- system.file("testdata", package = "platypus")
#   path <- file.path(testdata_path, "nested_dirs")
#   mode <- "nested_dirs"
#   classes <- 4
#   only_images <- FALSE
#   target_size <- c(2, 2)
#   grayscale <- FALSE
#   scale <- 1 / 255
#   batch_size <- 3
#   shuffle <- FALSE
#   subdirs <- c("images", "masks")
#
#   sample_generator <- segmentation_generator(path, mode, classes, only_images, target_size,
#                                              grayscale, scale, batch_size, shuffle, subdirs)
#   generator_output <- sample_generator()
#
#   set.seed(666)
#   sample_images <- array(sample(0:255, 3 * 2 * 2 * 3, replace = TRUE), dim = c(3, 2, 2, 3)) / 255
#   sample_masks <- array(sample(c(0, 255), 6 * 2 * 2 * 3, replace = TRUE),
#                         dim = c(6, 2, 2, 3))
#   sample_masks <- 1:3 %>% purrr::map(~ {
#     grayscale1 <- (0.2980392 * sample_masks[.x, , , 1] + 0.5843137 * sample_masks[.x, , , 2] + 0.1137255 * sample_masks[.x, , , 3])
#     grayscale2 <- (0.2980392 * sample_masks[.x + 3, , , 1] + 0.5843137 * sample_masks[.x + 3, , , 2] + 0.1137255 * sample_masks[.x + 3, , , 3])
#     grayscale <- reduce(list(grayscale1, grayscale2), `+`) / 255
#     array_reshape(grayscale, c(2, 2, 1))
#   }) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3))
#
#   expect_true(all(abs(sample_images - generator_output[[1]]) < 0.005))
#   expect_equal(sample_masks %>% keras::to_categorical(classes), generator_output[[2]])
# })
#
# test_that("segmentation_generator reads images and masks correctly from configuration CSV", {
#   testdata_path <- system.file("testdata", package = "platypus")
#   path <- file.path(tempdir(), "sample_config_file.csv")
#   nested_dirs <- list.dirs(file.path(testdata_path, "nested_dirs"), full.names  = TRUE, recursive = FALSE)
#   images_paths <- nested_dirs %>% purrr::map(~ list.files(file.path(.x, "images"), full.names  = TRUE))
#   masks_paths <- nested_dirs %>% purrr::map(~ paste0(list.files(file.path(.x, "masks"), full.names  = TRUE), collapse = ";"))
#   config_df <- data.frame(images = unlist(images_paths), masks = unlist(masks_paths))
#   readr::write_csv(config_df, path = path)
#
#   mode <- "config_file"
#   classes <- 4
#   only_images <- FALSE
#   target_size <- c(2, 2)
#   grayscale <- FALSE
#   scale <- 1 / 255
#   batch_size <- 3
#   shuffle <- FALSE
#
#   sample_generator <- segmentation_generator(path, mode, classes, only_images, target_size,
#                                              grayscale, scale, batch_size, shuffle, subdirs)
#   generator_output <- sample_generator()
#
#   set.seed(666)
#   sample_images <- array(sample(0:255, 3 * 2 * 2 * 3, replace = TRUE), dim = c(3, 2, 2, 3)) / 255
#   sample_masks <- array(sample(c(0, 255), 6 * 2 * 2 * 3, replace = TRUE),
#                         dim = c(6, 2, 2, 3))
#   sample_masks <- 1:3 %>% purrr::map(~ {
#     grayscale1 <- (0.2980392 * sample_masks[.x, , , 1] + 0.5843137 * sample_masks[.x, , , 2] + 0.1137255 * sample_masks[.x, , , 3])
#     grayscale2 <- (0.2980392 * sample_masks[.x + 3, , , 1] + 0.5843137 * sample_masks[.x + 3, , , 2] + 0.1137255 * sample_masks[.x + 3, , , 3])
#     grayscale <- reduce(list(grayscale1, grayscale2), `+`) / 255
#     array_reshape(grayscale, c(2, 2, 1))
#   }) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3))
#
#   expect_true(all(abs(sample_images - generator_output[[1]]) < 0.005))
#   expect_equal(sample_masks %>% keras::to_categorical(classes), generator_output[[2]])
# })
