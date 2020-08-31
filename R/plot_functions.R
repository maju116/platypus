#' Transforms pixel `array` into `data.frame` with raster data.
#' @description Transforms pixel `array` into `data.frame` with raster data.
#' @param xy_axis `x` and `y` image grid.
#' @param sample_image Pixel `array`.
#' @param grayscale Should images be plotted in grayscale.
#' @return  `data.frame` with raster data.
create_plot_data <- function(xy_axis, sample_image, grayscale) {
  if (grayscale) {
    cbind(xy_axis,
          gray = as.vector(t(sample_image[, , 1])) / max(sample_image[, , 1]))
  } else {
    cbind(xy_axis,
          r = as.vector(t(sample_image[, , 1])) / max(sample_image[, , 1]),
          g = as.vector(t(sample_image[, , 2])) / max(sample_image[, , 2]),
          b = as.vector(t(sample_image[, , 3])) / max(sample_image[, , 3]))
  }
}

#' Generates raster image.
#' @description Generates raster image.
#' @import ggplot2
#' @importFrom grDevices rgb gray
#' @param plot_data `data.frame` with `x`, `y` coordinates and color values.
#' @param grayscale Should images be plotted in grayscale.
#' @return  Raster image.
plot_raster <- function(plot_data, grayscale, fill_guide = FALSE) {
  base_plot <- if (grayscale) {
    ggplot(plot_data, aes(x, y, fill = gray(gray)))
  } else {
    ggplot(plot_data, aes(x, y, fill = rgb(r, g, b)))
  }
  base_plot + guides(fill = FALSE) + scale_fill_identity() +
    theme_void() + geom_raster(hjust = 0, vjust = 0)
}

#' Generates raster image with bounding boxes.
#' @description Generates raster image with bounding boxes.
#' @import ggplot2
#' @importFrom dplyr rename
#' @importFrom grDevices colorRampPalette
#' @import RColorBrewer
#' @param image_path Image filepath.
#' @param boxes `data.frame` with bounding boxes corresponding to the image.
#' @param labels Character vector containing class labels. For example \code{\link[platypus]{coco_labels}}.
#' @param correct_hw Logical. Should height/width rescaling of bounding boxes be applied.
#' @param target_size Image target size.
#' @param grayscale Should images be plotted in grayscale.
#' @return  Raster image with bounding boxes.
create_boxes_ggplot <- function(image_path, boxes, labels, correct_hw, target_size, grayscale) {
  sample_image <- image_load(image_path, target_size = target_size, grayscale = grayscale) %>%
    image_to_array()
  h <- dim(sample_image)[1]
  w <- dim(sample_image)[2]
  boxes <- if (correct_hw) correct_boxes(list(boxes), image_h = h, image_w = w)[[1]] else boxes
  boxes <- boxes %>% when(grayscale ~ mutate(., x = 0, y = 0, gray = 0),
                          ~ mutate(., x = 0, y = 0, r = 0, g = 0, b = 0))
  xy_axis <- expand.grid(1:w, h:1) %>% rename(x = Var1, y = Var2)
  plot_data <- create_plot_data(xy_axis, sample_image, grayscale)
  boxes_colors <- if (length(labels) > 8) {
    colorRampPalette(brewer.pal(8, "Set2"))(length(labels))
  } else {
    brewer.pal(length(labels), "Set2")
  }
  names(boxes_colors) <- labels
  plot_raster(plot_data, grayscale) +
    geom_rect(data = boxes, aes(xmin = xmin, ymin = h-ymin, xmax = xmax, ymax = h-ymax, colour = label),
              fill = NA, size = 1) +
    geom_label(data = boxes, aes(x = xmin, y = h-ymin, label = paste(label, round(p_obj * 100, 2), "%"), colour = label)) +
    theme(legend.position = "none") + scale_colour_manual(name = "boxes", values = boxes_colors)
}

#' Generates raster images with bounding boxes.
#' @description Generates raster images with bounding boxes.
#' @importFrom purrr walk2
#' @importFrom ggplot2 ggsave
#' @param images_paths Image filepaths.
#' @param boxes List of `data.frames` with bounding boxes corresponding to the images.
#' @param labels Character vector containing class labels. For example \code{\link[platypus]{coco_labels}}.
#' @param correct_hw Logical. Should height/width rescaling of bounding boxes be applied.
#' @param target_size Image target size.
#' @param grayscale Should images be plotted in grayscale.
#' @param save_dir Directory in which to save generated images.
#' @param plot_images Should images be plotted.
#' @return  Raster images with bounding boxes.
#' @export
plot_boxes <- function(images_paths, boxes, labels, correct_hw = TRUE,
                       target_size = NULL, grayscale = FALSE,
                       save_dir = NULL, plot_images = TRUE) {
  walk2(images_paths, boxes, ~ {
    p <- create_boxes_ggplot(.x, .y, labels, correct_hw, target_size, grayscale)
    if (plot_images) plot(p)
    if (!is.null(save_dir)) ggsave(filename = basename(.x), plot = p, path = save_dir)
  })
}

#' Generates raster image with segmentation masks.
#' @description Generates raster image with segmentation masks.
#' @import ggplot2
#' @importFrom dplyr rename left_join
#' @importFrom tibble as_tibble
#' @import RColorBrewer
#' @importFrom gridExtra grid.arrange
#' @param image_path Image filepath.
#' @param masks Predicted segmentation masks.
#' @param labels Character vector containing class labels. For example \code{\link[platypus]{voc_labels}}.
#' @param colormap Class color map. For example \code{\link[platypus]{voc_colormap}}.
#' @param target_size Image target size.
#' @param grayscale Should images be plotted in grayscale.
#' @param background_label Name of the background class label to filter out.
#' @return  Raster image with segmentation masks.
create_segmentation_map_ggplot <- function(image_path, masks, labels, colormap, target_size,
                                           grayscale, background_label) {
  colormap_df <- do.call("rbind", colormap) %>%
    as_tibble() %>%
    set_names(., c("r", "g", "b")) %>%
    mutate(label = labels)
  sample_image <- image_load(image_path, target_size = target_size, grayscale = grayscale) %>%
    image_to_array()
  h <- dim(sample_image)[1]
  w <- dim(sample_image)[2]
  xy_axis <- expand.grid(1:w, h:1) %>% rename(x = Var1, y = Var2)
  segmentation_mask <- masks %>%
    create_plot_data(xy_axis, ., FALSE) %>%
    left_join(colormap_df, by = c("r", "g", "b"))
  plot_data <- create_plot_data(xy_axis, sample_image, grayscale)
  true_plot <- plot_raster(plot_data, grayscale)
  predicted_mask <- plot_raster(segmentation_mask, FALSE)
  list(true_plot = true_plot, predicted_mask = predicted_mask)
}

#' Generates raster images with segmentation masks.
#' @description Generates raster images with segmentation masks.
#' @importFrom purrr walk2
#' @importFrom ggplot2 ggsave
#' @param images_paths Image filepaths.
#' @param binary_masks Predicted segmentation masks.
#' @param labels Character vector containing class labels. For example \code{\link[platypus]{voc_labels}}.
#' @param colormap Class color map. For example \code{\link[platypus]{voc_colormap}}.
#' @param grayscale Should images be plotted in grayscale.
#' @param save_dir Directory in which to save generated images.
#' @param plot_images Should images be plotted.
#' @param background_label Name of the background class label to filter out.
#' @return  Raster images with segmentation masks.
#' @export
plot_masks <- function(images_paths, masks, labels, colormap,
                       grayscale = FALSE, save_dir = NULL, plot_images = TRUE,
                       background_label = "background") {
  target_size <- dim(masks[[1]])[1:2]
  walk2(images_paths, masks, ~ {
    p <- create_segmentation_map_ggplot(.x, .y, labels, colormap, target_size,
                                        grayscale, background_label)
    if (plot_images) grid.arrange(p$true_plot, p$predicted_mask, nrow = 1)
    if (!is.null(save_dir)) ggsave(filename = basename(.x), plot = p, path = save_dir)
  })
}
