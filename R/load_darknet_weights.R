#' Loads `Yolo3 Darknet` (https://pjreddie.com/darknet/yolo/) weights into \code{\link[platypus]{yolo3}} model.
#' @description Loads `Yolo3 Darknet` (https://pjreddie.com/darknet/yolo/) weights into \code{\link[platypus]{yolo3}} model.
#' @import keras
#' @import progress
#' @importFrom purrr map_chr
#' @param model \code{\link[platypus]{yolo3}} model.
#' @param weights_path Filepath to weights file (You can download valid weights file from "https://pjreddie.com/media/files/yolov3.weights").
#' @return \code{\link[platypus]{yolo3}} model with pretrained weights.
#' @export
load_darknet_weights <- function(model, weights_path) {
  pb <- progress_bar$new(total = 75)
  submodels <- model$layers %>% map_chr(~ .$name) %>% .[c(2, 3, 6, 4, 7, 5, 8)]
  if (file.exists(weights_path)) {
    darknet_weights_file <-  file(weights_path, "rb")
  } else {
    stop("The 'weights_path' file does not exist!")
  }
  config <- readBin(darknet_weights_file, integer(), n = 5, endian = "little", size = 4)
  darknet_weights <- readBin(darknet_weights_file, double(), n = 62001757, endian = "little", size = 4)
  ind <- 1
  close(darknet_weights_file)
  for (submodel_name in submodels) {
    submodel <- model %>% get_layer(name = submodel_name)
    submodel_layer_names <- submodel$layers %>% map_chr(~ .$name)
    submodel_layer_len <- length(submodel_layer_names)
    for (i in 1:submodel_layer_len) {
      current_layer_name <- submodel_layer_names[i]
      if (grepl("conv2d", current_layer_name)) {
        pb$tick()
        conv_layer <- get_layer(submodel, current_layer_name)
        conv_layer_shape <- get_weights(conv_layer)
        batch_norm <- FALSE
        if (i + 1 < submodel_layer_len & grepl("batch_normalization", submodel_layer_names[i + 1])) {
          batch_norm <- TRUE
          batch_norm_layer <- get_layer(submodel, submodel_layer_names[i + 1])
          batch_norm_layer_shape <- dim(get_weights(batch_norm_layer)[[1]])
          beta <- darknet_weights[ind:(batch_norm_layer_shape + ind - 1)] %>% as.array()
          ind <- batch_norm_layer_shape + ind
          gamma <- darknet_weights[ind:(batch_norm_layer_shape + ind - 1)] %>% as.array()
          ind <- batch_norm_layer_shape + ind
          mean <- darknet_weights[ind:(batch_norm_layer_shape + ind - 1)] %>% as.array()
          ind <- batch_norm_layer_shape + ind
          var <- darknet_weights[ind:(batch_norm_layer_shape + ind - 1)] %>% as.array()
          ind <- batch_norm_layer_shape + ind
          set_weights(batch_norm_layer, list(gamma, beta, mean, var))
        } else {
          bias_shape <- dim(conv_layer_shape[[2]])
          bias <- darknet_weights[ind:(prod(bias_shape) + ind - 1)]
          ind <- prod(bias_shape) + ind
        }
        kernel_shape <- dim(conv_layer_shape[[1]])
        kernel <- darknet_weights[ind:(prod(kernel_shape) + ind - 1)]
        kernel <- aperm(array_reshape(kernel, rev(kernel_shape)), c(3, 4, 2, 1))
        ind <- prod(kernel_shape) + ind
        set_weights(conv_layer, if (batch_norm) list(kernel) else list(kernel, as.array(bias)))
      }
    }
  }
}
