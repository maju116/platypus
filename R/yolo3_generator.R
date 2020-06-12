read_annotations_from_xml <- function(annot_path, images_path) {
  list.files(annot_path, pattern = ".xml$", full.names = TRUE) %>%
    map(~ {
      data <- xmlParse(.x) %>%
        xmlToList()
      list("filename" = file.path(images_path, data$filename),
           "height" = as.numeric(data$size$height),
           "width" = as.numeric(data$size$width),
           "object" = data[names(data) == "object"] %>%
             map(~ cbind(name = .x$name, .x$bndbox %>% bind_cols())) %>%
             bind_rows() %>%
             mutate_at(c("xmin", "ymin", "xmax", "ymax"), ~ as.numeric(.)))
    })
}
