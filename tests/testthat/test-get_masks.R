context("get_masks")

test_that("get_masks correctly transform segmentation prediction into multi-class mask", {
  test_predictions <- array(
    c(0.1, 0.7, 0.8, 0.7, 0.05, 1, 0.05, 0.7, 0,
      0.1, 0.1, 0.1, 0.05, 0.8, 0, 0.9, 0.1, 0,
      0.1, 0.1, 0.05, 0.1, 0.05, 0, 0, 0.1, 1,
      0.7, 0.1, 0.05, 0.15, 0.1, 0, 0, 0.1, 0), dim = c(1, 3, 3, 4))
  test_colormap <- list(c(0, 0, 0), c(111, 111, 111),
                        c(222, 222, 222), c(255, 255, 255))
  expected_output <- list(array(c(255, 0, 0, 0, 111, 0, 111, 0, 222,
                             255, 0, 0, 0, 111, 0, 111, 0, 222,
                             255, 0, 0, 0, 111, 0, 111, 0, 222), dim = c(3, 3, 3)))
  true_output <- get_masks(test_predictions, test_colormap)

  expect_true(all.equal(expected_output[[1]], true_output[[1]], check.attributes = FALSE))
})
