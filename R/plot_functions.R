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

plot_boxes <- function(image_path, boxes, labels, n_class = 80) {
  sample_image <- image_load(image_path) %>% image_to_array()
  h <- dim(sample_image)[1]
  w <- dim(sample_image)[2]
  xy_axis <- expand.grid(1:w, h:1) %>% rename(x = Var1, y = Var2)
  plot_data <- create_plot_data(xy_axis, sample_image)
  p <- plot_rgb_raster(plot_data)
  boxes_data <- boxes %>% map_df(~ as.data.frame(t(.x))) %>%
    set_names(c("xmin", "ymin", "xmax", "ymax", "p_obj", paste0("class", 1:n_class))) %>%
    mutate(x = 0, y = 0, r = 0, g = 0, b = 0, label = labels)
  p + geom_rect(data = boxes_data, aes(xmin = xmin, ymin = h-ymin, xmax = xmax, ymax = h-ymax),
                fill = NA, colour = "red", size = 1) +
    geom_label(data = boxes_data, aes(x = xmin, y = h-ymin, label = label), colour = "red")
}
