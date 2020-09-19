context("dice_coeff")

test_that("dice_coeff calculates dice coefficient correctly", {
  test_y_true <- list(
    array(c(0, 0, 1, 0,
            0, 0, 1, 0,
            0, 0, 1, 0), dim = c(2, 2)),
    array(c(0, 1, 0, 0,
            0, 1, 0, 0,
            0, 1, 0, 0), dim = c(2, 2)),
    array(c(1, 0, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 0), dim = c(2, 2)),
    array(c(0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1), dim = c(2, 2))
  ) %>% abind::abind(along = 3)
  tensor_test_y_true <- keras::k_constant(test_y_true)
  test_y_pred <- list(
    array(c(0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1), dim = c(2, 2)),
    array(c(0, 1, 0, 0,
            0, 1, 0, 0,
            0, 1, 0, 0), dim = c(2, 2)),
    array(c(0, 0, 1, 0,
            0, 0, 1, 0,
            0, 0, 1, 0), dim = c(2, 2)),
    array(c(1, 0, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 0), dim = c(2, 2))
  ) %>% abind::abind(along = 3)
  tensor_test_y_pred <- keras::k_constant(test_y_pred)
  test_smooth <- 1

  expected_out <- (2 * sum(abs(test_y_true * test_y_pred)) + test_smooth) /
    (sum(test_y_true^2) + sum(test_y_pred^2) + test_smooth)
  true_out <- dice_coeff(test_y_true, test_y_pred, test_smooth)$numpy()
  expect_equal(expected_out, true_out)
})
