read_annotations_from_xml <- function(annot_paths, indices, images_path) {
  indices <- if (is.null(indices)) 1:length(paths) else indices
  annot_paths[indices] %>%
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

  grid_id <- true_boxes_anchors_iou$anchor_id %>% map_df(~ {
    tibble(
      grid_id = case_when(
        .x <= anchors_per_grid ~ 1,
        .x <= 2 * anchors_per_grid ~ 2,
        TRUE ~ 3
      )
    )
  })

  grid_dims <- grid_id$grid_id %>% map_df(~ {
    current_grid_id <- .x
    current_grid_w <- dim(true_grid[[current_grid_id]])[3]
    current_grid_h <- dim(true_grid[[current_grid_id]])[2]
    tibble(current_grid_w, current_grid_h)
  })

  bind_cols(true_boxes, true_boxes_anchors_iou, grid_id, grid_dims) %>%
    mutate(anchor_id_grid = anchor_id %% anchors_per_grid,
           anchor_id_grid = if_else(anchor_id_grid == 0, anchors_per_grid, anchor_id_grid))
}

get_true_boxes_from_annotations <- function(annotations, net_h, net_w, anchors, labels) {
  annotations %>% imap_dfr(~ {
    sample_id <- .y
    image_h <- .x$height
    image_w <- .x$width
    .x$object %>%
      rowwise() %>%
      mutate(label_id = which(label == labels),
             xmin = xmin / image_w * net_w,
             ymin = ymin / image_h * net_h,
             xmax = xmax / image_w * net_w,
             ymax = ymax / image_h * net_h
      ) %>%
      find_anchors_and_grids_for_true_boxes(anchors) %>%
      mutate(
        center_x = (xmin + xmax) / 2 / net_w * current_grid_w,
        center_y = (ymin + ymax) / 2 / net_h * current_grid_h,
        t_x = logit(center_x - floor(center_x)),
        t_y = logit(center_y - floor(center_y)),
        t_w = log((xmax - xmin) / anchor_w),
        t_h = log((ymax - ymin) / anchor_h),
        sample_id = sample_id
      ) %>%
      select(sample_id, center_x, center_y, t_x, t_y, t_w, t_h, anchor_id, anchor_id_grid, grid_id, label_id)
  })
}

yolo3_generator <- function(annot_path, images_path, net_h = 416, net_w = 416, grayscale = FALSE,
                            n_class = 80, anchors = coco_anchors, labels = coco_labels,
                            batch_size = 3, shuffle = TRUE) {
  anchors_per_grid <- length(anchors[[1]])
  downscale_grid <- c(32, 16, 8)
  annot_paths <- list.files(annot_path, pattern = ".xml$", full.names = TRUE)
  i <- 1
  function() {
    if (shuffle) {
      indices <- sample(1:length(annot_paths), size = batch_size)
    } else {
      indices <- c(i:min(i + batch_size - 1, length(annot_paths)))
      i <<- if (i + batch_size > length(annot_paths)) 1 else i + length(indices)
    }
  }
  true_grid <- downscale_grid %>% map(~ {
    generate_empty_grid(batch_size, net_h, net_w, .x, anchors_per_grid, n_class)
  })
  annotations <- read_annotations_from_xml(annot_paths, indices, images_path)
  true_boxes <- get_true_boxes_from_annotations(annotations, net_h, net_w, anchors, labels)
  for (i in 1:nrow(true_boxes)) {
    cbox <- true_boxes[i, ]
    cgrid_id <- cbox$grid_id
    csample_id <- cbox$sample_id
    crow <- floor(cbox$center_y)
    ccol <- floor(cbox$center_x)
    canchor_id_grid <- cbox$anchor_id_grid
    cbbox <- cbox %>% select(t_x, t_y, t_w, t_h) %>% as.numeric()
    clabel_id <- cbox$label_id
    true_grid[[cgrid_id]][csample_id, crow, ccol, canchor_id_grid, 1:4] <- cbbox
    true_grid[[cgrid_id]][csample_id, crow, ccol, canchor_id_grid, 5] <- 1
    true_grid[[cgrid_id]][csample_id, crow, ccol, canchor_id_grid, 5 + clabel_id] <- 1
  }
}
