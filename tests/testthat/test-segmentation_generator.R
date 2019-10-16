context("segmentation_generator")

test_that("segmentation_generator accepts only correct inputs.", {
  path <- ""
  correct_mode <- "nested_dirs"
  correct_only_images <- TRUE
  correct_target_size <- c(256, 256)
  correct_grayscale <- TRUE
  scale <- 1 / 255
  batch_size <- 1
  correct_shuffle <- FALSE
  subdirs = c("images", "masks")
  column_sep = ";"

  incorrect_mode_1 <- "config"
  incorrect_mode_2 <- 4
  incorrect_only_images <- NA
  incorrect_target_size_1 <- c(111, 256)
  incorrect_target_size_1 <- c(256, 111)
  incorrect_grayscale <- NA
  incorrect_shuffle <- NA

  expect_output(segmentation_generator(path, correct_mode, correct_only_images, correct_target_size,
                                       correct_grayscale, scale, batch_size, correct_shuffle,
                                       subdirs, column_sep))
  expect_error(segmentation_generator(path, incorrect_mode_1, correct_only_images, correct_target_size,
                                       correct_grayscale, scale, batch_size, correct_shuffle,
                                       subdirs, column_sep))
  expect_error(segmentation_generator(path, incorrect_mode_2, correct_only_images, correct_target_size,
                                       correct_grayscale, scale, batch_size, correct_shuffle,
                                       subdirs, column_sep))
  expect_error(segmentation_generator(path, correct_mode, incorrect_only_images, correct_target_size,
                                       correct_grayscale, scale, batch_size, correct_shuffle,
                                       subdirs, column_sep))
  expect_error(segmentation_generator(path, correct_mode, correct_only_images, incorrect_target_size_1,
                                       correct_grayscale, scale, batch_size, correct_shuffle,
                                       subdirs, column_sep))
  expect_error(segmentation_generator(path, correct_mode, correct_only_images, incorrect_target_size_2,
                                       correct_grayscale, scale, batch_size, correct_shuffle,
                                       subdirs, column_sep))
  expect_error(segmentation_generator(path, correct_mode, correct_only_images, correct_target_size,
                                      incorrect_grayscale, scale, batch_size, correct_shuffle,
                                       subdirs, column_sep))
  expect_error(segmentation_generator(path, correct_mode, correct_only_images, correct_target_size,
                                       correct_grayscale, scale, batch_size, incorrect_grayscale,
                                       subdirs, column_sep))
})

test_that("segmentation_generator reads images and masks correctly from directory", {
  x
})
