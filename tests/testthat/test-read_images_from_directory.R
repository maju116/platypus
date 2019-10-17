context("read_images_from_directory")

test_that("read_images_from_directory reads images and masks correctly", {
  testdata_path <- system.file("testdata", package = "platypus")
  nested_dirs <- list.dirs(file.path(testdata_path, "nested_dirs"), full.names  = TRUE, recursive = FALSE)
  images_paths <- nested_dirs %>% purrr::map(~ list.files(file.path(.x, "images"), full.names  = TRUE))
  masks_paths <- nested_dirs %>% purrr::map(~ list.files(file.path(.x, "masks"), full.names  = TRUE))
  indices <- 2:3
  target_size <- c(2, 2)
  grayscale_images <- FALSE
  grayscale_masks <- TRUE
  scale <- 1 / 255

  images <- read_images_from_directory(images_paths, indices, target_size, grayscale_images, scale)
  masks <- read_images_from_directory(masks_paths, indices, target_size, grayscale_masks, scale)

  set.seed(666)
  sample_images <- array(sample(0:255, 3 * 2 * 2 * 3, replace = TRUE), dim = c(3, 2, 2, 3)) / 255
  sample_masks <- array(sample(c(0, 255), 6 * 2 * 2 * 3, replace = TRUE),
                        dim = c(6, 2, 2, 3))
  sample_masks <- 1:3 %>% purrr::map(~ {
    grayscale1 <- (0.2980392 * sample_masks[.x, , , 1] + 0.5843137 * sample_masks[.x, , , 2] + 0.1137255 * sample_masks[.x, , , 3])
    grayscale2 <- (0.2980392 * sample_masks[.x + 3, , , 1] + 0.5843137 * sample_masks[.x + 3, , , 2] + 0.1137255 * sample_masks[.x + 3, , , 3])
    grayscale <- reduce(list(grayscale1, grayscale2), `+`) / 255
    array_reshape(grayscale, c(2, 2, 1))
  }) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3))

  expect_true(all(abs(sample_images[indices, , , ] - images) < 0.005))
  expect_true(all(abs(sample_masks[indices, , , , drop = FALSE] - masks) < 0.005))
})
