#' Transforms pixel `array` into `data.frame` with raster data.
#' @description Transforms pixel `array` into `data.frame` with raster data.
#' @param xy_axis `x` and `y` image grid.
#' @param sample_image Pixel `array`.
#' @return  `data.frame` with raster data.
create_plot_data <- function(xy_axis, sample_image){
  cbind(xy_axis,
        r = as.vector(t(sample_image[, , 1])) / max(sample_image[, , 1]),
        g = as.vector(t(sample_image[, , 2])) / max(sample_image[, , 2]),
        b = as.vector(t(sample_image[, , 3])) / max(sample_image[, , 3]))
}

#' Generates raster image.
#' @description Generates raster image.
#' @import ggplot2
#' @importFrom grDevices rgb
#' @param plot_data `data.frame` with `x`, `y` coordinates and color values.
#' @return  Raster image.
plot_rgb_raster <- function(plot_data){
  ggplot(plot_data, aes(x, y, fill = rgb(r, g, b))) +
    guides(fill = FALSE) + scale_fill_identity() + theme_void() +
    geom_raster(hjust = 0, vjust = 0)
}

#' Generates raster image with bounding boxes.
#' @description Generates raster image with bounding boxes.
#' @import ggplot2
#' @importFrom dplyr rename
#' @param image_path Image filepath.
#' @param boxes `data.frame` with bounding boxes corresponding to the image.
#' @param correct_hw Logical. Should height/width rescaling of bounding boxes be applied.
#' @param target_size Image target size.
#' @return  Raster image with bounding boxes.
plot_boxes_ggplot <- function(image_path, boxes, correct_hw, target_size) {
  sample_image <- image_load(image_path, target_size = target_size) %>%
    image_to_array()
  h <- dim(sample_image)[1]
  w <- dim(sample_image)[2]
  boxes <- if (correct_hw) correct_boxes(list(boxes), image_h = h, image_w = w)[[1]] else boxes
  boxes <- boxes %>% mutate(x = 0, y = 0, r = 0, g = 0, b = 0)
  xy_axis <- expand.grid(1:w, h:1) %>% rename(x = Var1, y = Var2)
  plot_data <- create_plot_data(xy_axis, sample_image)
  p <- plot_rgb_raster(plot_data) +
    geom_rect(data = boxes, aes(xmin = xmin, ymin = h-ymin, xmax = xmax, ymax = h-ymax, colour = label),
              fill = NA, size = 1) +
    geom_label(data = boxes, aes(x = xmin, y = h-ymin, label = label, colour = label)) +
    theme(legend.position = "none")
  plot(p)
}

#' Generates raster images with bounding boxes.
#' @description Generates raster images with bounding boxes.
#' @importFrom purrr walk2
#' @param images_paths Image filepaths.
#' @param boxes List of `data.frames` with bounding boxes corresponding to the images.
#' @param correct_hw Logical. Should height/width rescaling of bounding boxes be applied.
#' @param target_size Image target size.
#' @return  Raster images with bounding boxes.
#' @export
plot_boxes <- function(images_paths, boxes, correct_hw = TRUE, target_size = NULL) {
  walk2(images_paths, boxes, ~ plot_boxes_ggplot(.x, .y, correct_hw, target_size))
}
