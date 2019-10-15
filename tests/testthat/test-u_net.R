context("u_net")

test_that("u_net accepts only correct inputs.", {
  correct_input_shape <- c(256, 256, 3)
  correct_blocks <- 3
  correct_classes <- 5
  correct_filters <- 12
  correct_dropout <- 0.567
  correct_batch_normalization <- TRUE

  incorrect_input_shape_1 <- c(111, 256, 3)
  incorrect_input_shape_2 <- c(256, 111, 3)
  incorrect_input_shape_3 <- c(256, 256, -1)
  incorrect_input_shape_4 <- c(256, 256)
  incorrect_blocks <- -3
  incorrect_classes <- 1
  incorrect_filters <- -12
  incorrect_dropout_1 <- 1.567
  incorrect_dropout_2 <- -1.567
  incorrect_batch_normalization <- NA

  expect_silent(u_net(correct_input_shape, correct_blocks, correct_classes, correct_filters,
                      correct_dropout, correct_batch_normalization))
  expect_error(u_net(incorrect_input_shape_1, correct_blocks, correct_classes, correct_filters,
                      correct_dropout, correct_batch_normalization))
  expect_error(u_net(incorrect_input_shape_2, correct_blocks, correct_classes, correct_filters,
                      correct_dropout, correct_batch_normalization))
  expect_error(u_net(incorrect_input_shape_3, correct_blocks, correct_classes, correct_filters,
                      correct_dropout, correct_batch_normalization))
  expect_error(u_net(incorrect_input_shape_4, correct_blocks, correct_classes, correct_filters,
                      correct_dropout, correct_batch_normalization))
  expect_error(u_net(correct_input_shape, incorrect_blocks, correct_classes, correct_filters,
                      correct_dropout, correct_batch_normalization))
  expect_error(u_net(correct_input_shape, correct_blocks, incorrect_classes, correct_filters,
                      correct_dropout, correct_batch_normalization))
  expect_error(u_net(correct_input_shape, correct_blocks, correct_classes, incorrect_filters,
                      correct_dropout, correct_batch_normalization))
  expect_error(u_net(correct_input_shape, correct_blocks, correct_classes, correct_filters,
                      incorrect_dropout_1, correct_batch_normalization))
  expect_error(u_net(correct_input_shape, correct_blocks, correct_classes, correct_filters,
                      incorrect_dropout_2, correct_batch_normalization))
  expect_error(u_net(correct_input_shape, correct_blocks, correct_classes, correct_filters,
                     correct_dropout, incorrect_batch_normalization))
})

test_that("u_net creates correct network architecture.", {
  input_shape <- c(256, 256, 3)
  blocks_1 <- 3
  blocks_2 <- 5
  classes <- 5
  filters <- 12
  dropout <- 0.567
  batch_normalization_1 <- TRUE
  batch_normalization_2 <- FALSE

  model1 <- u_net(input_shape, blocks_1, classes, filters,
                 dropout, batch_normalization_1)
  model2 <- u_net(input_shape, blocks_2, classes, filters,
                  dropout, batch_normalization_2)

  len_layers_out_1 <- length(model1$layers)
  expected_len_layers_out_1 <- 1 + blocks_1 * 8 + 6 + blocks_1 * 9 + 1
  len_layers_out_2 <- length(model2$layers)
  expected_len_layers_out_2 <- 1 + blocks_2 * 6 + 4 + blocks_2 * 7 + 1

  expect_equal(len_layers_out_1, expected_len_layers_out_1)
  expect_equal(len_layers_out_2, expected_len_layers_out_2)
})
