#' Reads images from directory.
#' @description Reads images from directory.
#' @import keras
#' @importFrom abind abind
#' @importFrom purrr map
#' @param paths Images directories.
#' @param indices Indices of selected images. If `NULL` all images in `paths` will be selected.
#' @param target_size Images size (height, width). Default to `c(256, 256)`.
#' @param grayscale Boolean, whether to load the image as grayscale.
#' @param scale Scaling factor for images pixel values. Default to `1 / 255`.
#' @export
read_images_from_directory <- function(paths, indices = NULL, target_size = c(256, 256), grayscale = FALSE, scale = 1 / 255) {
  indices <- if (is.null(indices)) 1:length(paths) else indices
  map(paths[indices], ~ {
    image_to_array(image_load(.x, grayscale = grayscale, target_size = target_size)) * scale
  }) %>% abind(along = 4) %>% aperm(c(4, 1, 2, 3))
}

#' Generates batches of data (images and masks). The data will be looped over (in batches).
#' @description Generates batches of data (images and masks). The data will be looped over (in batches).
#' @param path Images and masks directory.
#' @param only_images Should generator read only images (e.g. on train set for predictions).
#' @param target_size Images / mask size (height, width). Default to `c(256, 256)`.
#' @param grayscale Boolean, whether to load the image as grayscale.
#' @param scale Scaling factor for images pixel values. Default to `1 / 255`.
#' @param batch_size Batch size.
#' @param shuffle Should data be shuffled.
#' @param subdirs Vector of two characters containing names of subdirectories with images and masks.
#' @export
segmentation_generator <- function(path, only_images = FALSE, target_size = c(256, 256), grayscale = FALSE, scale = 1 / 255,
                                   batch_size = 32, shuffle = TRUE, subdirs = c("images", "masks")) {
  images_paths <- list.files(file.path(path, subdirs[1]), full.names  = TRUE)
  if (!only_images) masks_paths <- list.files(file.path(path, subdirs[2]), full.names = TRUE)
  # all(basename(images_paths) == basename(masks_paths))
  print(paste0(length(images_paths), " images", if (!only_images) " with corresponding masks", " detected!"))
  i <- 1
  function() {
    if (shuffle) {
      indices <- sample(1:length(images_paths), size = batch_size)
    } else {
      indices <- c(i:min(i + batch_size - 1, length(images_paths)))
      i <<- if (i + batch_size >= length(images_paths)) 1 else i + length(indices)
    }
    images <- read_images_from_directory(images_paths, indices = indices, target_size = target_size,
                                         grayscale = grayscale, scale = scale)
    if (!only_images) masks <- read_images_from_directory(masks_paths, indices = indices, target_size = target_size,
                                        grayscale = TRUE, scale = 1 / 255) %>% to_categorical()
    if (!only_images) list(images, masks) else list(images)
  }
}
