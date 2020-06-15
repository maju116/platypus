create_plot_data <- function(xy_axis, sample_image){
  cbind(xy_axis,
        r = as.vector(t(sample_image[, , 1])) / 255,
        g = as.vector(t(sample_image[, , 2])) / 255,
        b = as.vector(t(sample_image[, , 3])) / 255)
}

plot_rgb_raster <- function(plot_data){
  ggplot(plot_data, aes(x, y, fill = rgb(r, g, b))) +
    guides(fill = FALSE) + scale_fill_identity() + theme_void() +
    geom_raster(hjust = 0, vjust = 0)
}

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

plot_boxes <- function(images_paths, boxes, correct_hw = TRUE, target_size = NULL) {
  walk2(images_paths, boxes, ~ plot_boxes_ggplot(.x, .y, correct_hw, target_size))
}
