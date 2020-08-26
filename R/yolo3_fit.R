#' Fits `YOLOv3` model using data generator.
#' @description Fits `YOLOv3` model using data generator.
#' @param model `YOLOv3` Model.
#' @param generator `YOLOv3` Data generator.
#' @param epochs Number of epochs.
#' @param steps_per_epoch Steps per epoch.
#' @param validation_generator Validation data generator.
#' @param validation_steps_per_epoch Validation steps per epoch.
#' @import progress
#' @export
yolo3_fit_generator <- function(model, generator, epochs, steps_per_epoch,
                                validation_generator = NULL,
                                validation_steps_per_epoch = NULL) {
  results <- NULL
  val_results <- NULL
  for (epoch in 1:epochs) {
    pb_format <- paste("Epoch:", epoch, "[:bar] :percent eta: :eta")
    pb <- progress_bar$new(total = steps_per_epoch, format = pb_format)
    epoch_steps_all <- map_dfr(1:steps_per_epoch, function(x) {
      current_batch <- generator()
      metrics <- train_on_batch(model, current_batch[[1]], current_batch[[2]]) %>%
        set_names("loss", "grid1_loss", "grid2_loss", "grid3_loss",
                  "grid1_avg_IoU", "grid2_avg_IoU", "grid3_avg_IoU") %>%
        as.list() %>% as_tibble()
      pb$tick()
      metrics
    })
    epoch_steps_last <- slice(epoch_steps_all, steps_per_epoch)
    results <- bind_rows(results, epoch_steps_last)
    print(paste("Epoch:", epoch))
    print(paste(colnames(epoch_steps_last), round(epoch_steps_last, 3),
                sep = ": ", collapse = ", "))
    if (!is.null(validation_generator)) {
      val_epoch_steps_all <- map_dfr(1:validation_steps_per_epoch, function(x) {
        val_current_batch <- validation_generator()
        val_metrics <- test_on_batch(model, val_current_batch[[1]], val_current_batch[[2]]) %>%
          set_names("val_loss", "val_grid1_loss", "val_grid2_loss", "val_grid3_loss",
                    "val_grid1_avg_IoU", "val_grid2_avg_IoU", "val_grid3_avg_IoU") %>%
          as.list() %>% as_tibble()
        val_metrics
      })
      val_epoch_steps_last <- slice(val_epoch_steps_all, validation_steps_per_epoch)
      val_results <- bind_rows(val_results, val_epoch_steps_last)
      print(paste(colnames(val_epoch_steps_last), round(val_epoch_steps_last, 3),
                  sep = ": ", collapse = ", "))
    }
  }
  results <- results %>%
    mutate(epoch = 1:epochs) %>%
    when(!is.null(validation_generator) ~ bind_cols(., val_results), ~ .)
  return(results)
}
