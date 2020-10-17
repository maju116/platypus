context("yolo3")

test_that("yolo3 accepts only correct inputs", {
  correct_net_h <- 256
  correct_net_w <- 256
  correct_grayscale <- FALSE
  correct_classes <- 5
  correct_anchors <- coco_anchors

  incorrect_net_h <- 111
  incorrect_net_w <- 255
  incorrect_classes <- 0
  incorrect_anchors_1 <- -12
  incorrect_anchors_2 <- list(1, 2, 3, 4)
  incorrect_anchors_3 <- list(list(1, 2), list(3), list(2, 5))

  expect_silent(yolo3(correct_net_h, correct_net_w, correct_grayscale, correct_classes,
                      correct_anchors))
  expect_error(yolo3(incorrect_net_h, correct_net_w, correct_grayscale, correct_classes,
                      correct_anchors))
  expect_error(yolo3(correct_net_h, incorrect_net_w, correct_grayscale, correct_classes,
                      correct_anchors))
  expect_error(yolo3(correct_net_h, correct_net_w, correct_grayscale, incorrect_classes,
                      correct_anchors))
  expect_error(yolo3(correct_net_h, correct_net_w, correct_grayscale, correct_classes,
                     incorrect_anchors_1))
  expect_error(yolo3(correct_net_h, correct_net_w, correct_grayscale, correct_classes,
                     incorrect_anchors_2))
  expect_error(yolo3(correct_net_h, correct_net_w, correct_grayscale, correct_classes,
                     incorrect_anchors_3))
})

test_that("yolo3 creates correct network architecture.", {
  net_h_1 <- 256
  net_w_1 <- 256
  net_h_2 <- 416
  net_w_2 <- 1024
  grayscale <- FALSE
  classes_1 <- 5
  classes_2 <- 10
  anchors_1 <- coco_anchors
  anchors_2 <- list(list(c(1, 2), c(1, 2), c(1, 2), c(1, 2)),
                    list(c(1, 2), c(1, 2), c(1, 2), c(1, 2)),
                    list(c(1, 2), c(1, 2), c(1, 2), c(1, 2)))

  model_1 <- yolo3(net_h_1, net_w_1, grayscale, classes_1, anchors_1)
  model_2 <- yolo3(net_h_2, net_w_2, grayscale, classes_2, anchors_2)

  model_1_output_shape <- model_1$output %>% purrr::map(~ .$get_shape()$as_list() %>% unlist())
  model_2_output_shape <- model_2$output %>% purrr::map(~ .$get_shape()$as_list() %>% unlist())

  expect_equal(model_1_output_shape, list(c(8, 8, 3, 10), c(16, 16, 3, 10), c(32, 32, 3, 10)))
  expect_equal(model_2_output_shape, list(c(13, 32, 4, 15), c(26, 64, 4, 15), c(52, 128, 4, 15)))
})
