context("metric_dice_coeff")

test_that("metric_dice_coeff is a proper python function", {
  expect_equal(c("python.builtin.function", "python.builtin.object"),
               class(metric_dice_coeff()))
})
