#' Fits `YOLOv3` model using data generator.
#' @description Fits `YOLOv3` model using data generator.
#' @param model `YOLOv3` Model.
#' @param generator `YOLOv3` Data generator.
#' @param epochs Number of epochs.
#' @param steps_per_epoch Steps per epoch.
#' @import progress
#' @export
yolo3_fit_generator <- function(model, generator, epochs, steps_per_epoch) {
  results <- NULL
  for (epoch in 1:epochs) {
    pb_format <- paste("Epoch:", epoch, "[:bar] :percent eta: :eta")
    pb <- progress_bar$new(total = steps_per_epoch, format = pb_format)
    epoch_steps_all <- map_dfr(1:steps_per_epoch, function(x) {
      current_batch <- generator()
      metrics <- train_on_batch(model, current_batch[[1]], current_batch[[2]])
      metrics <- tibble(
        .loss = metrics[[1]],
        .grid1_loss = metrics[[2]],
        .grid2_loss = metrics[[3]],
        .grid3_loss = metrics[[4]],
        .grid1_avg_IoU = metrics[[5]],
        .grid2_avg_IoU = metrics[[6]],
        .grid3_avg_IoU = metrics[[7]]
      )
      pb$tick()
      metrics
    })
    epoch_steps_last <- slice(epoch_steps_all, steps_per_epoch)
    results <- bind_rows(results, epoch_steps_last)
    print(paste(colnames(epoch_steps_last), round(epoch_steps_last, 3),
                sep = ": ", collapse = ", "))
  }
  return(results)
}
