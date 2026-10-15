# A vector field as a picture.
#
# A convention rather than a decision, which is why it is not in `drawing_style()` and why
# the test is against the engine rather than against a palette: the only way to get a
# different picture from the same field is to measure the angle from somewhere else, and
# that is not a choice either package gets to make differently.
#
# Reachable from R only since `predict(type = "channels")`, which is why this is `#103`'s
# file and not `#154`'s: `#154` deferred `flow_to_colours()` for a reason that had already
# expired - `#160` shipped and `separation_flows()` is exported - and the one that actually
# held was that no flow field could get into R at all.

test_that("a flow field is the engine's picture of it, pixel for pixel", {
  skip_if_no_engine()
  set.seed(7)
  for (shape in list(c(2L, 6L, 7L), c(2L, 1L, 5L), c(2L, 9L, 2L))) {
    field <- array(stats::rnorm(prod(shape)), dim = shape)
    reticulate::py_run_string(sprintf(
      "import numpy as np\nfield = np.array(%s).reshape(%d, %d, %d, order='F')",
      paste0("[", paste(format(as.vector(field), digits = 17), collapse = ","), "]"),
      shape[[1]], shape[[2]], shape[[3]]
    ))
    theirs <- reticulate::py_run_string("import pyplatypus
out = pyplatypus.flow_to_colours(field)")$out
    expect_equal(as.vector(flow_to_colours(field)), as.vector(theirs),
                 info = paste(shape, collapse = "x"))
  }
})

test_that("a zero vector is black, which is what the background is", {
  field <- array(0, dim = c(2, 3, 3))
  field[1, 1, 1] <- 1
  shown <- flow_to_colours(field)
  expect_true(all(shown[2, 2, ] == 0L))
  expect_false(all(shown[1, 1, ] == 0L))
})

test_that("a field with no vector anywhere is all black rather than a division by zero", {
  # The branch the engine guards with `longest > 0`. An untrained flow head predicts near
  # zero everywhere, so this is the first picture somebody draws, not an edge case.
  expect_true(all(flow_to_colours(array(0, dim = c(2, 4, 4))) == 0L))
})

test_that("something that is not a two-channel field is refused", {
  expect_error(flow_to_colours(array(0, dim = c(3, 4, 4))), "2 x height x width")
  expect_error(flow_to_colours(matrix(0, 4, 4)), "2 x height x width")
})
