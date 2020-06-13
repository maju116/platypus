read_annotations_from_xml <- function(annot_path, images_path) {
  list.files(annot_path, pattern = ".xml$", full.names = TRUE) %>%
    map(~ {
      data <- xmlParse(.x) %>%
        xmlToList()
      list("filename" = file.path(images_path, data$filename),
           "height" = as.numeric(data$size$height),
           "width" = as.numeric(data$size$width),
           "object" = data[names(data) == "object"] %>%
             map(~ cbind(label = .x$name, .x$bndbox %>% bind_cols())) %>%
             bind_rows() %>%
             mutate_at(c("xmin", "ymin", "xmax", "ymax"), ~ as.numeric(.)))
    })
}

generate_empty_grid <- function(batch_size, net_h, net_w, downscale, anchors_per_grid, n_class) {
  array(data = 0, dim = c(batch_size, net_h / downscale, net_w / downscale,
                          anchors_per_grid, 5 + n_class))
}

find_anchors_and_grids_for_true_boxes <- function(true_boxes, anchors) {
  anchors_boxes <- unlist(anchors, recursive = FALSE) %>% map_df(~ {
    tibble(xmin = 0, ymin = 0, xmax = .x[1], ymax = .[2])
  }) %>% mutate(anchor_id = row_number())

  shiftedd_boxes <- true_boxes %>%
    mutate(xmax = xmax - xmin, ymax = ymax - ymin, xmin = 0, ymin = 0) %>%
    select(-label, -label_id)

  true_boxes_anchors_iou <- shiftedd_boxes %>%
    pmap_df(function(xmin, ymin, xmax, ymax) {
      true_box <- c(xmin, ymin, xmax, ymax)
      pmap_df(anchors_boxes, function(xmin, ymin, xmax, ymax, anchor_id) {
        current_anchor <- c(xmin, ymin, xmax, ymax)
        current_anchor_id <- anchor_id
        current_iou <- intersection_over_union(true_box, current_anchor)
        tibble(anchor_id = current_anchor_id, iou = current_iou,
               anchor_w = xmax, anchor_h = ymax)
      }) %>% filter(iou == max(iou))
    }) %>% select(-iou)

  grid_nr <- true_boxes_anchors_iou$anchor_id %>% map_df(~ {
    tibble(
      grid_nr = case_when(
        .x <= anchors_per_grid ~ 1,
        .x <= 2 * anchors_per_grid ~ 2,
        TRUE ~ 3
      )
    )
  })

  grid_dims <- grid_nr$grid_nr %>% map_df(~ {
    current_grid_nr <- .x
    current_grid_w <- dim(true_grid[[current_grid_nr]])[3]
    current_grid_h <- dim(true_grid[[current_grid_nr]])[2]
    tibble(current_grid_w, current_grid_h)
  })

  bind_cols(true_boxes, true_boxes_anchors_iou, grid_nr, grid_dims)
}

yolo3_generator <- function(true_boxes, net_h = 416, net_w = 416, grayscale = FALSE,
                            n_class = 80, anchors = coco_anchors,
                            batch_size = 3) {
  anchors_per_grid <- length(anchors[[1]])
  downscale_grid <- c(32, 16, 8)
  true_grid <- downscale_grid %>% map(~ {
    generate_empty_grid(batch_size, net_h, net_w, .x, anchors_per_grid, n_class)
  })

  true_boxes_grid <- true_boxes %>%
    find_anchors_and_grids_for_true_boxes(anchors) %>%
    mutate(
      center_x = (xmin + xmax) / 2 / net_w * current_grid_w, # net_w -> img_w ?
      center_y = (ymin + ymax) / 2 / net_h * current_grid_h, # net_h -> img_h ?
      w = log((xmax - xmin) / anchor_w),
      h = log((ymax - ymin) / anchor_h)
    )
}
