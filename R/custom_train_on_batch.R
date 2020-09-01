as_generator.function <- function (x) {
  python_path <- system.file("python", package = "keras")
  tools <- reticulate::import_from_path("kerastools", path = python_path)
  iter <- reticulate::py_iterator(function() {
    elem <- keras_array(x())
    if (length(elem) == 1)
      elem[[2]] <- list()
    do.call(reticulate::tuple, elem)
  })
  tools$generator$iter_generator(iter)
}

#' Fits model using data generator.
#' @description Fits model using data generator.
#' @import progress
#' @importFrom purrr map_dfr
#' @importFrom dplyr summarise_all
#' @param metric_names Metric names.
#' @param model Model.
#' @param generator Data generator.
#' @param epochs Number of epochs.
#' @param steps_per_epoch Steps per epoch.
#' @param validation_generator Validation data generator.
#' @param validation_steps_per_epoch Validation steps per epoch.
#' @param model_filepath Path to save the model.
#' @export
custom_fit_generator <- function(metric_names, model, generator, epochs, steps_per_epoch,
                                validation_generator = NULL,
                                validation_steps_per_epoch = NULL,
                                model_filepath = NULL) {
  results <- NULL
  val_results <- NULL
  for (epoch in 1:epochs) {
    pb_format <- paste("Epoch:", epoch, "[:bar] :percent eta: :eta")
    pb <- progress_bar$new(total = steps_per_epoch, format = pb_format)
    epoch_steps_all <- map_dfr(1:steps_per_epoch, function(x) {
      current_batch <- generator()
      metrics <- train_on_batch(model, current_batch[[1]], current_batch[[2]]) %>%
        set_names(metric_names) %>%
        as.list() %>% as_tibble()
      pb$tick()
      metrics
    })
    epoch_steps_mean <- epoch_steps_all %>%
      summarise_all(mean)
    results <- bind_rows(results, epoch_steps_mean)
    cat(paste("Epoch:", epoch, "\n"))
    cat(paste(paste(colnames(epoch_steps_mean), round(epoch_steps_mean, 3),
                    sep = ": ", collapse = ", "), "\n"))
    if (!is.null(validation_generator)) {
      val_epoch_steps_all <- map_dfr(1:validation_steps_per_epoch, function(x) {
        val_current_batch <- validation_generator()
        val_metrics <- test_on_batch(model, val_current_batch[[1]], val_current_batch[[2]]) %>%
          set_names(paste0("val_", metric_names)) %>%
          as.list() %>% as_tibble()
        val_metrics
      })
      val_epoch_steps_mean <- val_epoch_steps_all %>%
        summarise_all(mean)
      val_results <- bind_rows(val_results, val_epoch_steps_mean)
      cat(paste(paste(colnames(val_epoch_steps_mean), round(val_epoch_steps_mean, 3),
                      sep = ": ", collapse = ", "), "\n"))
    }
    if (!is.null(model_filepath)) {
      save_model_hdf5(model, model_filepath)
    }
  }
  results <- results %>%
    mutate(epoch = 1:epochs) %>%
    when(!is.null(validation_generator) ~ bind_cols(., val_results), ~ .)
  return(results)
}

#' Calculates predictions on new samples using data generator.
#' @description Calculates predictions on new samples using data generator.
#' @param model Model.
#' @param generator Data generator.
#' @param steps Steps in epoch.
#' @import progress
#' @importFrom stats predict
#' @export
custom_predict_generator <- function(model, generator, steps) {
  pb_format <- "[:bar] :percent eta: :eta"
  pb <- progress_bar$new(total = steps, format = pb_format)
  map(1:steps, function(x) {
    current_batch <- generator()
    predictions <- predict(model, current_batch[[1]])
    pb$tick()
    predictions
  })
}
