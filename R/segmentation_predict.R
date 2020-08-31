#' Unites binary masks into multi-class mask.
#' @description Unites binary masks into multi-class mask.
#' @param masks Segmentation masks.
#' @param colormap Class color map. For example \code{\link[platypus]{voc_colormap}}.
#' @export
unite_binary_masks <- function(masks, colormap) {
  colormap %>% imap(~ {
    current_color <- .x
    color_index <- .y
    current_color %>% map(~ {
      masks[ , , color_index, drop = FALSE] * .x
    }) %>% abind(along = 3)
  }) %>% reduce(`+`)
}

#' Transforms `U-Net` predictions into valid segmentation map.
#' @description Transforms `U-Net` predictions into valid segmentation map.
#' @importFrom purrr map
#' @param preds \code{\link[platypus]{u_net}} model predictions.
#' @param colormap Class color map. For example \code{\link[platypus]{voc_colormap}}.
#' @export
get_masks <- function(preds, colormap) {
  1:dim(preds)[1] %>% map(~ {
    current_pred <- preds[.x, , , ]
    current_pred %>% apply(1:2, which.max) %>%
      `-`(1) %>%
      to_categorical(length(colormap)) %>%
      unite_binary_masks(colormap)
  })
}
