box_jaccard_distance <- function(box1_w, box1_h, box2_w, box2_h) {
  top <- min(box1_w, box2_w) * min(box1_h, box2_h)
  bottom <- box1_w * box1_h + box2_w * box2_h - top
  1 - top / bottom
}

initialize_anchors <- function(annot_df, total_anchors) {
  initial_anchors <- annot_df %>% sample_n(1) %>% select(-label)

  for (anchor in 2:total_anchors) {
    min_distance <- annot_df %>% select(-label) %>%
      pmap_dbl(function(box_w, box_h) {
        current_box <- c(box_w, box_h)
        initial_anchors %>% pmap_dbl(function(box_w, box_h, anchor_id) {
          current_anchor <- c(box_w, box_h)
          box_jaccard_distance(current_box[1], current_box[2],
                               current_anchor[1], current_anchor[2])
        }) %>% min()
      })
    new_anchor_id <- sample(1:nrow(annot_df), 1, prob = min_distance / sum(min_distance))
    initial_anchors <- initial_anchors %>% bind_rows(
      annot_df[new_anchor_id, ] %>% select(-label)
    )
  }
  initial_anchors %>%
    mutate(anchor_id = 1:total_anchors)
}

generate_anchors <- function(anchors_per_grid, annot_paths,
                             labels, net_h, net_w, n_iter = 10) {
  total_anchors <- anchors_per_grid * 3
  annotations <- read_annotations_from_xml(annot_paths, NULL, "", labels)
  annot_df <- annotations %>% map_df(~ {
    image_h <- .x$height
    image_w <- .x$width
    .x$object %>%
      mutate(box_w = round((xmax - xmin) / image_w * net_w),
             box_h = round((ymax - ymin) / image_h * net_h)
      )
  }) %>% select(box_w, box_h, label)
  print(annot_df %>% count(label))
  base_plot <- ggplot(annot_df, aes(box_w, box_h, color = label)) + geom_point() + theme_bw()
  initial_anchors <- initialize_anchors(annot_df, total_anchors)
  for(i in 1:n_iter) {
    best_anchors <- annot_df %>% select(-label) %>%
      pmap_dbl(function(box_w, box_h) {
        current_box <- c(box_w, box_h)
        initial_anchors %>% pmap_dbl(function(box_w, box_h, anchor_id) {
          current_anchor <- c(box_w, box_h)
          box_jaccard_distance(current_box[1], current_box[2],
                               current_anchor[1], current_anchor[2])
        }) %>% which.min()
      })
    new_anchors <- annot_df %>% select(-label) %>%
      mutate(anchor_id = best_anchors) %>%
      group_by(anchor_id) %>%
      summarise(box_w = median(box_w), box_h = median(box_h), .groups = 'drop') %>%
      ungroup() %>%
      arrange(anchor_id)
    if (identical(new_anchors, initial_anchors)) break
    initial_anchors <- new_anchors
  }
  plot(base_plot + geom_point(data = new_anchors, color = "red", shape = 23))
  new_anchors %>% mutate(area = box_w * box_h) %>%
    arrange(desc(area)) %>% select(-area) %>%
    mutate(anchor_id = 1:total_anchors)
}
