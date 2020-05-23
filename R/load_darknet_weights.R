read_darknet_weights <- function(model, weights_path = "development/yolov3.weights") {
  darknet_weights_file <- file(weights_path, "rb")
  config <- readBin(darknet_weights_file, integer(), n = 3, endian = "little", size = 4)
  if ((config[1] * 10 + config[2]) >= 2 & config[1] < 1000 & config[2] < 1000) {
    readBin(darknet_weights_file, integer(), n = 1, endian = "little", size = 8)
  } else {
    readBin(darknet_weights_file, integer(), n = 1, endian = "little", size = 4)
  }
  darknet_weights <- readBin(darknet_weights_file, double(), n = 62001757, endian = "little", size = 4)
  layer_ids <- yolo3_config %>% map_dbl(~ .$id) %>% c(80, 81, 92, 93) %>% sort()
  ind <- 1
  for (layer_id in layer_ids) {
    conv_layer <- get_layer(model, paste0("conv2d_", layer_id))
    conv_layer_shape <- get_weights(conv_layer)
    if (!(layer_id %in% c(81, 93, 105))) {
      batch_norm_layer <- get_layer(model, paste0("batch_norm_", layer_id))
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
    }
    if (length(conv_layer_shape) > 1) {
      bias_shape <- dim(conv_layer_shape[[2]])
      bias <- darknet_weights[ind:(prod(bias_shape) + ind - 1)]
      ind <- prod(bias_shape) + ind
      kernel_shape <- dim(conv_layer_shape[[1]])
      kernel <- darknet_weights[ind:(prod(kernel_shape) + ind - 1)]
      kernel <- aperm(array_reshape(kernel, rev(kernel_shape)), c(3, 4, 2, 1))
      ind <- prod(kernel_shape) + ind
      set_weights(conv_layer, list(kernel, as.array(bias)))
    } else {
      kernel_shape <- dim(conv_layer_shape[[1]])
      kernel <- darknet_weights[ind:(prod(kernel_shape) + ind - 1)]
      kernel <- aperm(array_reshape(kernel, rev(kernel_shape)), c(3, 4, 2, 1))
      ind <- prod(kernel_shape) + ind
      set_weights(conv_layer, list(kernel))
    }
    print(paste("Layer", layer_id, "loaded!"))
  }
}

read_darknet_weights <- function(model, weights_path = "development/yolov3.weights") {
  submodels <- model$layers %>% map_chr(~ .$name) %>% .[c(2, 3, 6, 4, 7, 5, 8)]
  darknet_weights_file <- file(weights_path, "rb")
  config <- readBin(darknet_weights_file, integer(), n = 5, endian = "little", size = 4)
  darknet_weights <- readBin(darknet_weights_file, double(), n = 62001757, endian = "little", size = 4)
  ind <- 1
  for (submodel_name in submodels) {
    submodel <- model %>% get_layer(name = submodel_name)
    submodel_layer_names <- submodel$layers %>% map_chr(~ .$name)
    submodel_layer_len <- length(submodel_layer_names)
    for (i in 1:submodel_layer_len) {
      current_layer_name <- submodel_layer_names[i]
      if (grepl("conv2d", current_layer_name)) {
        print(paste("Inserting weights to :", current_layer_name))
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
