#' Fits `segmentation` model using data generator.
#' @description Fits `segmentation` model using data generator.
#' @param model `segmentation` Model.
#' @param generator `segmentation` Data generator.
#' @param epochs Number of epochs.
#' @param steps_per_epoch Steps per epoch.
#' @param validation_generator Validation data generator.
#' @param validation_steps_per_epoch Validation steps per epoch.
#' @param model_filepath Path to save the model.
#' @import progress
#' @export
segmentation_fit_generator <- function(model, generator, epochs, steps_per_epoch,
                                validation_generator = NULL,
                                validation_steps_per_epoch = NULL,
                                model_filepath = NULL) {
  metric_names <- c("loss", "dice_coef")
  custom_fit_generator(metric_names, model, generator, epochs, steps_per_epoch,
                       validation_generator,
                       validation_steps_per_epoch,
                       model_filepath)
}

#' Calculates predictions on new samples using `segmentation` data generator.
#' @description Calculates predictions on new samples using `segmentation` data generator.
#' @param model `segmentation` Model.
#' @param generator `segmentation` Data generator.
#' @param steps Steps in epoch.
#' @import progress
#' @export
segmentation_predict_generator <- function(model, generator, steps) {
  predictions <- custom_predict_generator(model, generator, steps) %>%
    abind(along = 1)
  predictions
}
