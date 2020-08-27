#' Fits `YOLOv3` model using data generator.
#' @description Fits `YOLOv3` model using data generator.
#' @param model `YOLOv3` Model.
#' @param generator `YOLOv3` Data generator.
#' @param epochs Number of epochs.
#' @param steps_per_epoch Steps per epoch.
#' @param validation_generator Validation data generator.
#' @param validation_steps_per_epoch Validation steps per epoch.
#' @param model_filepath Path to save the model.
#' @import progress
#' @export
yolo3_fit_generator <- function(model, generator, epochs, steps_per_epoch,
                                validation_generator = NULL,
                                validation_steps_per_epoch = NULL,
                                model_filepath = NULL) {
  metric_names <- c("loss", "grid1_loss", "grid2_loss", "grid3_loss",
                    "grid1_avg_IoU", "grid2_avg_IoU", "grid3_avg_IoU")
  custom_fit_generator(metric_names, model, generator, epochs, steps_per_epoch,
                       validation_generator,
                       validation_steps_per_epoch,
                       model_filepath)
}
