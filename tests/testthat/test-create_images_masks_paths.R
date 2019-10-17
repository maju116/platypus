context("create_images_masks_paths")

test_that("create_images_masks_paths creates paths correctly", {
  testdata_path <- system.file("testdata", package = "platypus")
  path_1 <- file.path(testdata_path, "dir")
  path_2 <- file.path(testdata_path, "nested_dirs")
  path_3 <- file.path(tempdir(), "sample_config_file.csv")
  images_paths_1 <- list.files(file.path(path_1, "images"), full.names  = TRUE) %>% as.list()
  masks_paths_1 <- list.files(file.path(path_1, "masks"), full.names = TRUE) %>% as.list()
  nested_dirs <- list.dirs(file.path(testdata_path, "nested_dirs"), full.names  = TRUE, recursive = FALSE)
  images_paths_2 <- nested_dirs %>% purrr::map(~ list.files(file.path(.x, "images"), full.names  = TRUE))
  masks_paths_2 <- nested_dirs %>% purrr::map(~ list.files(file.path(.x, "masks"), full.names  = TRUE))
  masks_paths_3 <- nested_dirs %>% purrr::map(~ paste0(list.files(file.path(.x, "masks"), full.names  = TRUE), collapse = ";"))
  config_df <- data.frame(images = unlist(images_paths_2), masks = unlist(masks_paths_3))
  readr::write_csv(config_df, path = path_3)

  mode_1 <- "dir"
  mode_2 <- "nested_dirs"
  mode_3 <- "config_file"
  only_images <- FALSE
  subdirs <- c("images", "masks")
  column_sep = ";"

  result_1 <- create_images_masks_paths(path_1, mode_1, only_images, subdirs, column_sep)
  result_2 <- create_images_masks_paths(path_2, mode_2, only_images, subdirs, column_sep)
  result_3 <- create_images_masks_paths(path_3, mode_3, only_images, subdirs, column_sep)

  expect_equal(result_1, list(images_paths = images_paths_1, masks_paths = masks_paths_1))
  expect_equal(result_2, list(images_paths = images_paths_2, masks_paths = masks_paths_2))
  expect_equal(result_3, list(images_paths = images_paths_2, masks_paths = masks_paths_2))
})
