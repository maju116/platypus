context("split_masks_into_binary")

test_that("split_masks_into_binary splits multi-class mask into binary masks correctly", {
  test_mask <- array(c(255, 0, 0, 0, 111, 0, 111, 0, 222,
                       255, 0, 0, 0, 111, 0, 111, 0, 222,
                       255, 0, 0, 0, 111, 0, 111, 0, 222), dim = c(3, 3, 3))
  test_colormap_1 <- list(c(0, 0, 0), c(111, 111, 111),
                        c(222, 222, 222), c(255, 255, 255))
  test_colormap_2 <- list(c(0, 0, 0), c(222, 222, 222),
                          c(255, 255, 255), c(111, 111, 111))
  expected_out_1 <- list(
    matrix(c(0, 1, 1, 1, 0, 1, 0, 1, 0), nrow = 3),
    matrix(c(0, 0, 0, 0, 1, 0, 1, 0, 0), nrow = 3),
    matrix(c(0, 0, 0, 0, 0, 0, 0, 0, 1), nrow = 3),
    matrix(c(1, 0, 0, 0, 0, 0, 0, 0, 0), nrow = 3)
  )
  expected_out_2 <- list(
    matrix(c(0, 1, 1, 1, 0, 1, 0, 1, 0), nrow = 3),
    matrix(c(0, 0, 0, 0, 0, 0, 0, 0, 1), nrow = 3),
    matrix(c(1, 0, 0, 0, 0, 0, 0, 0, 0), nrow = 3),
    matrix(c(0, 0, 0, 0, 1, 0, 1, 0, 0), nrow = 3)
  )
  true_out_1 <- split_masks_into_binary(test_mask, test_colormap_1)
  true_out_2 <- split_masks_into_binary(test_mask, test_colormap_2)
  expect_equal(expected_out_1, true_out_1)
  expect_equal(expected_out_2, true_out_2)
})
