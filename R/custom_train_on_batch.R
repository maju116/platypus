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
#' @import ggplot2
#' @import gridExtra
#' @importFrom purrr map_dfr
#' @importFrom dplyr summarise_all
#' @importFrom keras generator_next
#' @param metric_names Metric names.
#' @param model Model.
#' @param generator Data generator.
#' @param epochs Number of epochs.
#' @param steps_per_epoch Steps per epoch.
#' @param validation_generator Validation data generator.
#' @param validation_steps_per_epoch Validation steps per epoch.
#' @param model_filepath Path to save the model.
#' @param save_best_only If set to `TRUE` model will be saved only if selected `metric` was better than in previous epochs.
#' @param monitor Metric selected to be monitored.
#' @param monitor_choose_best How to compare new monitor with previous ones. One of `c("min", "max")`
#' @export
custom_fit_generator <- function(metric_names, model, generator, epochs, steps_per_epoch,
                                validation_generator = NULL,
                                validation_steps_per_epoch = NULL,
                                model_filepath = NULL,
                                save_best_only = TRUE,
                                monitor = "loss",
                                monitor_choose_best = "min") {
  results <- NULL
  val_results <- NULL
  for (epoch in 1:epochs) {
    pb_format <- paste("Epoch:", epoch, "[:bar] :percent eta: :eta")
    pb <- progress_bar$new(total = steps_per_epoch, format = pb_format)
    epoch_steps_all <- map_dfr(1:steps_per_epoch, function(x) {
      current_batch <- generator_next(generator)
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
        val_current_batch <- generator_next(validation_generator)
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
      if (save_best_only & epoch > 1) {
        compare_fun <- if (monitor_choose_best == "min") `<=` else `>=`
        if (monitor %in% metric_names) {
          current_monitor <- epoch_steps_mean %>% pull(monitor)
          old_monitor <- results[1:(epoch - 1), ] %>% pull(monitor)
        } else {
          current_monitor <- val_epoch_steps_mean %>% pull(monitor)
          old_monitor <- val_results[1:(epoch - 1), ] %>% pull(monitor)
        }
        if (all(compare_fun(current_monitor, old_monitor))) {
          print(paste("Saving model to", model_filepath))
          save_model_hdf5(model, model_filepath)
        }
      } else {
        print(paste("Saving model to", model_filepath))
        save_model_hdf5(model, model_filepath)
      }
    }
  }
  results <- results %>%
    mutate(epoch = 1:epochs) %>%
    when(!is.null(validation_generator) ~ bind_cols(., val_results), ~ .)
  metric_names %>% map(~ {
    metric_name <- .x
    val_metric_name <- paste0("val_", metric_name)
    p <- ggplot(results, aes(x = epoch)) + theme_bw() +
      geom_line(aes(y = !!sym(metric_name), color = metric_name))
    p <- if (!is.null(validation_generator)) {
      p + geom_line(aes(y = !!sym(val_metric_name), color = val_metric_name))
    } else {
      p
    }
  }) %>% grid.arrange(grobs = ., ncol = 1)
  return(results)
}

#' Fits `YOLOv3` model using data generator.
#' @description Fits `YOLOv3` model using data generator.
#' @param model `YOLOv3` Model.
#' @param generator `YOLOv3` Data generator.
#' @param epochs Number of epochs.
#' @param steps_per_epoch Steps per epoch.
#' @param validation_generator Validation data generator.
#' @param validation_steps_per_epoch Validation steps per epoch.
#' @param model_filepath Path to save the model.
#' @param save_best_only If set to `TRUE` model will be saved only if selected `metric` was better than in previous epochs.
#' @param monitor Metric selected to be monitored.
#' @import progress
#' @export
yolo3_fit_generator <- function(model, generator, epochs, steps_per_epoch,
                                validation_generator = NULL,
                                validation_steps_per_epoch = NULL,
                                model_filepath = NULL,
                                save_best_only = TRUE,
                                monitor = "loss") {
  metric_names <- c("loss", "grid1_loss", "grid2_loss", "grid3_loss",
                    "grid1_avg_IoU", "grid2_avg_IoU", "grid3_avg_IoU")
  monitor_choose_best <- if (grepl("loss", monitor)) "min" else "max"
  if (!is.null(model_filepath) & save_best_only & !(monitor %in% c(metric_names, paste0("val_", metric_names)))) {
    stop("Incorrect metric choosen as monitor!")
  }
  custom_fit_generator(metric_names, model, generator, epochs, steps_per_epoch,
                       validation_generator,
                       validation_steps_per_epoch,
                       model_filepath,
                       save_best_only,
                       monitor,
                       monitor_choose_best)
}

#' Calculates predictions on new samples using data generator.
#' @description Calculates predictions on new samples using data generator.
#' @importFrom keras generator_next
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
    current_batch <- generator_next(generator)
    predictions <- predict(model, current_batch[[1]])
    pb$tick()
    predictions
  })
}
