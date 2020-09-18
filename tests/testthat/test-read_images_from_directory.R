context("read_images_from_directory")

test_that("read_images_from_directory reads images and masks correctly from nested_dirs", {
  testdata_path <- system.file("testdata", package = "platypus")
  nested_dirs <- list.dirs(file.path(testdata_path, "nested_dirs"), full.names  = TRUE, recursive = FALSE)
  images_paths <- nested_dirs %>% purrr::map(~ list.files(file.path(.x, "images"), full.names  = TRUE))
  masks_paths <- nested_dirs %>% purrr::map(~ list.files(file.path(.x, "masks"), full.names  = TRUE))
  indices <- NULL
  target_size <- c(2, 2)
  grayscale <- FALSE
  scale <- 1 / 255
  colormap <- list(c(0, 0, 0), c(111, 111, 111),
                   c(222, 222, 222), c(255, 255, 255))

  true_images <- read_images_from_directory(images_paths, indices, target_size, grayscale, scale, NULL)
  true_masks <- read_images_from_directory(masks_paths, indices, target_size, grayscale, scale, colormap)

  expected_images <- list(
    array(c(255, 111, 0, 222,
            255, 111, 0, 222,
            255, 111, 0, 222), dim = c(2, 2, 3)),
    array(c(255, 111, 222, 0,
            255, 111, 222, 0,
            255, 111, 222, 0), dim = c(2, 2, 3)),
    array(c(222, 111, 0, 255,
            222, 111, 0, 255,
            222, 111, 0, 255), dim = c(2, 2, 3))
  ) %>% abind::abind(along = 4) %>% aperm(c(4, 1, 2, 3)) %>% `/`(255)

  expected_masks <- list(
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

  expect_equal(expected_images, true_images)
  expect_equal(expected_masks, true_masks)
})
