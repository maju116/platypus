segmentation_generator <- function(path, target_height, target_width, grayscale = FALSE, scale = 1/255, batch_size, shuffle = FALSE, subdirs = c("images", "masks")) {
  images_paths <- list.files(file.path(path, subdirs[1]), full.names  = TRUE)
  masks_paths <- list.files(file.path(path, subdirs[2]), full.names = TRUE)
  all(basename(images_paths) == basename(masks_paths))
  i <- 1
  function() {
    if (shuffle) {
      indices <- sample(1:length(images_paths), size = batch_size)
    } else {
      if (i + batch_size >= length(images_paths))
        i <<- 1
      indices <- c(i:min(i + batch_size - 1, nrow(images_paths)))
      i <<- i + length(indices)
    }
    images <- map(images_paths[indices], ~ {
      image_array_resize(image_to_array(image_load(.x, grayscale = grayscale)), target_height, target_width) * scale
    }) %>% abind(along = 4) %>% aperm(c(4, 1, 2, 3))

    masks <- map(masks_paths[indices], ~ {
      image_array_resize(image_to_array(image_load(.x, grayscale = TRUE)), target_height, target_width) * scale
    }) %>% abind(along = 4) %>% aperm(c(4, 1, 2, 3)) %>% to_categorical()

    list(images, masks)
  }
}
