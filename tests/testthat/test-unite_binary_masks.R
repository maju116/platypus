context("unite_binary_masks")

test_that("unite_binary_masks unites binary masks correctly", {
  test_binary_mask <- array(
    c(0, 1, 1, 1, 0, 1, 0, 1, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1,
      1, 0, 0, 0, 0, 0, 0, 0, 0), dim = c(3, 3, 4))
  test_colormap <- list(c(0, 0, 0), c(111, 111, 111),
                        c(222, 222, 222), c(255, 255, 255))
  expected_output <- array(c(255, 0, 0, 0, 111, 0, 111, 0, 222,
                             255, 0, 0, 0, 111, 0, 111, 0, 222,
                             255, 0, 0, 0, 111, 0, 111, 0, 222), dim = c(3, 3, 3))
  true_output <- unite_binary_masks(test_binary_mask, test_colormap)

  expect_true(all.equal(expected_output, true_output, check.attributes = FALSE))
})
