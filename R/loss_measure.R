#' Function for calculating dice coefficient.
#' @description Function for calculating dice coefficient.
#' @import keras
#' @param y_true Tensor of true probabilities.
#' @param y_pred Tensor of predicted probabilities.
#' @param smooth Smoothing Factor.
#' @return Dice coefficient.
dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  intersection <- k_sum(k_abs(y_true * y_pred))
  (2 * intersection + smooth) / (k_sum(k_square(y_true)) + k_sum(k_square(y_pred)) + smooth)
}

#' Dice coefficient metric.
#' @description Dice coefficient metric.
#' @import keras
#' @return Dice coefficient metric.
#' @export
metric_dice_coef <- custom_metric("dice_coeff", function(y_true, y_pred) {
  dice_coef(y_true, y_pred, smooth = 1.0)
})

#' Dice loss.
#' @description Dice loss.
#' @import keras
#' @return Dice loss.
#' @export
dice_loss <- custom_metric("dice_loss", function(y_true, y_pred) {
  1 - dice_coef(y_true, y_pred, smooth = 1.0)
})
