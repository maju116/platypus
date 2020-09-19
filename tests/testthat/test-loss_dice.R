context("loss_dice")

test_that("loss_dice is a proper python function", {
  expect_equal(c("python.builtin.function", "python.builtin.object"),
               class(loss_dice()))
})
