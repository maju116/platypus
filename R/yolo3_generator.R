#' Reads bounding box annotations from JSON files. Each XML has to be in `labelme JSON` format.
#' @description Reads bounding box annotations from JSON files. Each JSON has to be in `labelme JSON` format.
#' @import jsonlite
#' @param annot_paths List to XML annotations filepaths.
#' @param indices Indices specifying which files to read. If `NULL` all files are loaded.
#' @param images_path Path to directory with images.
#' @param labels Character vector containing class labels. For example \code{\link[platypus]{coco_labels}}.
#' @return List of bounding box coordinates, heights, widths and image filepaths.
#' @export
read_annotations_from_labelme <- function(annot_paths, indices, images_path, labels) {
  indices <- if (is.null(indices)) 1:length(annot_paths) else indices
  annot_paths[indices] %>%
    map(~ {
      data <- fromJSON(.x)
      list("filename" = file.path(images_path, basename(data$imagePath)),
           "height" = as.numeric(data$imageHeight),
           "width" = as.numeric(data$imageWidth),
           "object" = tibble(
             label = data$shapes$label
           ) %>%
             bind_cols(data$shapes$points %>%
                         map_df(~ {
                           as.data.frame(cbind(.x[1, , drop = FALSE], .x[2, , drop = FALSE])) %>%
                             transmute(xmin = min(V1, V3), ymin = min(V2, V4),
                                       xmax = max(V1, V3), ymax = max(V2, V4))
                         })) %>%
             filter(label %in% labels))
    })
}

#' Reads bounding box annotations from XML files. Each XML has to be in `PASCAL VOC XML` format.
#' @description Reads bounding box annotations from XML files. Each XML has to be in `PASCAL VOC XML` format.
#' @import XML
#' @param annot_paths List to XML annotations filepaths.
#' @param indices Indices specifying which files to read. If `NULL` all files are loaded.
#' @param images_path Path to directory with images.
#' @param labels Character vector containing class labels. For example \code{\link[platypus]{coco_labels}}.
#' @return List of bounding box coordinates, heights, widths and image filepaths.
#' @export
read_annotations_from_xml <- function(annot_paths, indices, images_path, labels) {
  indices <- if (is.null(indices)) 1:length(annot_paths) else indices
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
             mutate_at(c("xmin", "ymin", "xmax", "ymax"), ~ as.numeric(.)) %>%
             filter(xmin < xmax & ymin < ymax & label %in% labels))
    })
}

#' Creates empty output `Yolo3` grid.
#' @description Creates empty output `Yolo3` grid.
#' @param batch_size Batch size.
#' @param net_h Input layer height from trained \code{\link[platypus]{yolo3}} model. Must be divisible by `32`.
#' @param net_w Input layer width from trained \code{\link[platypus]{yolo3}} model. Must be divisible by `32`.
#' @param downscale Integer specifying how to downscale `net_h` and `net_w`. For `Yolo3` output grids equal do `32`, `16` and `8`.
#' @param anchors_per_grid Number of anchors/boxes per one output grid.
#' @param n_class Number of prediction classes.
#' @return Empty output `Yolo3` grid.
generate_empty_grid <- function(batch_size, net_h, net_w, downscale, anchors_per_grid, n_class) {
  array(data = 0, dim = c(batch_size, net_h / downscale, net_w / downscale,
                          anchors_per_grid, 5 + n_class))
}

#' Finds best anchors and output grid coordinates for true bounding box coordinates.
#' @description Finds best anchors and output grid coordinates for true bounding box coordinates.
#' @importFrom purrr map_df pmap_df
#' @importFrom dplyr mutate row_number select filter bind_cols if_else
#' @importFrom tibble tibble
#' @param true_boxes True bounding box coordinates.
#' @param anchors Prediction anchors. For exact format check \code{\link[platypus]{coco_anchors}}.
#' @param true_grid `Yolo3` output grids.
#' @return `data.frame` with best anchors and output grid coordinates for true bounding box coordinates.
find_anchors_and_grids_for_true_boxes <- function(true_boxes, anchors, true_grid) {
  anchors_per_grid <- length(anchors[[1]])
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

#' Calculates true bounding box coordinates from annotations.
#' @description Calculates true bounding box coordinates from annotations.
#' @importFrom purrr imap_dfr
#' @param annotations Annotations.
#' @param anchors Prediction anchors. For exact format check \code{\link[platypus]{coco_anchors}}.
#' @param labels Character vector containing class labels. For example \code{\link[platypus]{coco_labels}}.
#' @param true_grid `Yolo3` output grids.
#' @return `data.frame` with best anchors and output grid coordinates for true bounding box coordinates.
get_true_boxes_from_annotations <- function(annotations, anchors, labels, true_grid) {
  annotations %>% imap_dfr(~ {
    sample_id <- .y
    image_h <- .x$height
    image_w <- .x$width
    .x$object %>%
      rowwise() %>%
      mutate(label_id = which(label == labels),
             xmin = xmin / image_w,
             ymin = ymin / image_h,
             xmax = xmax / image_w,
             ymax = ymax / image_h
      ) %>%
      find_anchors_and_grids_for_true_boxes(anchors, true_grid) %>%
      mutate(
        center_x = (xmin + xmax) / 2 * current_grid_w,
        center_y = (ymin + ymax) / 2 * current_grid_h,
        t_x = logit(center_x - floor(center_x)),
        t_y = logit(center_y - floor(center_y)),
        t_w = log((xmax - xmin) / anchor_w),
        t_h = log((ymax - ymin) / anchor_h),
        sample_id = sample_id
      ) %>%
      select(sample_id, center_x, center_y, t_x, t_y, t_w, t_h, anchor_id, anchor_id_grid, grid_id, label_id)
  })
}

#' Generates batches of data (images and box coordinates/scores). The data will be looped over (in batches).
#' @description Generates batches of data (images and box coordinates/scores). The data will be looped over (in batches).
#' @import keras
#' @importFrom abind abind
#' @importFrom purrr map
#' @param annot_path Annotations directory.
#' @param images_path Images directory.
#' @param only_images Should generator read only images (e.g. on train set for predictions).
#' @param net_h Input layer height. Must be divisible by `32`.
#' @param net_w Input layer width. Must be divisible by `32`.
#' @param annot_format Annotations format. One of `pascal_voc`, `labelme`.
#' @param grayscale Defines input layer color channels -  `1` if `TRUE`, `3` if `FALSE`.
#' @param scale Scaling factor for images pixel values. Default to `1 / 255`.
#' @param anchors Prediction anchors. For exact format check \code{\link[platypus]{coco_anchors}}.
#' @param labels Character vector containing class labels. For example \code{\link[platypus]{coco_labels}}.
#' @param batch_size Batch size.
#' @param shuffle Should data be shuffled.
#' @export
yolo3_generator <- function(annot_path, images_path, only_images = FALSE,
                            net_h = 416, net_w = 416, annot_format = "pascal_voc",
                            grayscale = FALSE, scale = 1 / 255,
                            anchors = coco_anchors, labels = coco_labels,
                            batch_size = 32, shuffle = TRUE) {
  yolo3_generator_check(only_images, net_h, net_w, annot_format,
                        grayscale, anchors, labels)
  n_class <- length(labels)
  anchors_per_grid <- length(anchors[[1]])
  downscale_grid <- c(32, 16, 8)
  annot_ext <- if (annot_format == "pascal_voc") ".xml$" else ".json$"
  annot_paths <- list.files(annot_path, pattern = annot_ext, full.names = TRUE)
  cat(paste0(length(annot_paths), " images", if (!only_images) " with corresponding annotations", " detected!\n"))
  cat(paste0("Set 'steps_per_epoch' to: ", ceiling(length(annot_paths) / batch_size), "\n"))
  i <- 1
  as_generator.function(function() {
    if (shuffle) {
      indices <- sample(1:length(annot_paths), size = batch_size)
    } else {
      indices <- c(i:min(i + batch_size - 1, length(annot_paths)))
      i <<- if (i + batch_size > length(annot_paths)) 1 else i + length(indices)
    }
    annotations <- if (annot_format == "pascal_voc") {
      read_annotations_from_xml(annot_paths, indices, images_path, labels)
    } else {
      read_annotations_from_labelme(annot_paths, indices, images_path, labels)
    }
    if (!only_images) {
      true_grid <- downscale_grid %>% map(~ {
        generate_empty_grid(batch_size, net_h, net_w, .x, anchors_per_grid, n_class)
      })
      true_boxes <- get_true_boxes_from_annotations(annotations, anchors, labels, true_grid)
      for (i in 1:nrow(true_boxes)) {
        cbox <- true_boxes[i, ]
        cgrid_id <- cbox$grid_id
        csample_id <- cbox$sample_id
        crow <- floor(cbox$center_y) + 1
        ccol <- floor(cbox$center_x) + 1
        canchor_id_grid <- cbox$anchor_id_grid
        cbbox <- cbox %>% select(t_x, t_y, t_w, t_h) %>% as.numeric()
        clabel_id <- cbox$label_id
        true_grid[[cgrid_id]][csample_id, crow, ccol, canchor_id_grid, 1:4] <- cbbox
        true_grid[[cgrid_id]][csample_id, crow, ccol, canchor_id_grid, 5] <- 1
        true_grid[[cgrid_id]][csample_id, crow, ccol, canchor_id_grid, 5 + clabel_id] <- 1
      }
    }
    images_paths <- annotations %>% map(~ .$filename)
    images <- images_paths %>% map(~ image_to_array(image_load(.x, grayscale = grayscale,
                                                               target_size = c(net_h, net_w))) * scale) %>%
      abind(along = 4) %>% aperm(c(4, 1, 2, 3))
    if (!only_images) list(images, true_grid) else list(images)
  })
}
