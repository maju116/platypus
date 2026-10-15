# How big each object is, in units somebody can use.
#
# `mask_volume()` makes the argument for one class: a voxel count means nothing outside one
# scanner. This is the same argument per object, and it is the question `#103` opens with -
# *how many nuclei, and how big is each one* - which the published weights' card cannot
# answer.

two_objects <- function() {
  labels <- matrix(0L, nrow = 8, ncol = 8)
  labels[2:4, 2:4] <- 1L   # 9 pixels
  labels[6:7, 6:8] <- 2L   # 6 pixels
  labels
}

test_that("one row per object, counted", {
  sizes <- instance_sizes(two_objects())
  expect_identical(sizes$object, c(1L, 2L))
  expect_identical(sizes$voxels, c(9L, 6L))
  # No size column without a spacing, rather than a count pretending to be a measurement.
  expect_identical(names(sizes), c("object", "voxels"))
})

test_that("a spacing gives an area whose unit is in the column's name", {
  sizes <- instance_sizes(two_objects(), spacing = c(0.5, 0.5))
  # Arithmetic by hand: 9 pixels of 0.5 x 0.5 mm is 9 * 0.25 = 2.25 mm^2.
  expect_equal(sizes$area_mm2, c(2.25, 1.5))
  expect_identical(names(sizes), c("object", "voxels", "area_mm2"))
})

test_that("a volume is measured in millilitres, as mask_volume reports them", {
  labels <- array(0L, dim = c(4, 4, 4))
  labels[1:2, 1:2, 1:2] <- 1L   # 8 voxels
  sizes <- instance_sizes(labels, spacing = c(2, 2, 5))
  # 8 voxels of 2 x 2 x 5 mm is 160 mm^3, which is 0.16 ml. The same conversion
  # `mask_volume()` makes, and asserted against the same arithmetic rather than against it.
  expect_equal(sizes$volume_ml, 0.16)
  expect_identical(names(sizes), c("object", "voxels", "volume_ml"))
  expect_equal(mask_volume(array(ifelse(labels == 1L, 2L, 1L), dim = dim(labels)),
                           spacing = c(2, 2, 5)),
               sizes$volume_ml)
})

test_that("a gap in the object numbers is three objects and not four", {
  # A decoding that dropped object 3 for being too small leaves 1, 2 and 4. Counting rows is
  # then the only honest answer; counting up to the maximum would invent one.
  labels <- matrix(0L, nrow = 6, ncol = 6)
  labels[1, 1] <- 1L
  labels[2, 2] <- 2L
  labels[3, 3] <- 4L
  sizes <- instance_sizes(labels)
  expect_identical(sizes$object, c(1L, 2L, 4L))
  expect_identical(nrow(sizes), 3L)
})

test_that("an empty map is no rows rather than one row of nothing", {
  sizes <- instance_sizes(matrix(0L, nrow = 4, ncol = 4))
  expect_identical(nrow(sizes), 0L)
  expect_identical(names(sizes), c("object", "voxels"))
})

test_that("a spacing of the wrong length is refused, naming both numbers", {
  expect_error(instance_sizes(two_objects(), spacing = c(1, 1, 1)),
               "must be 2 positive numbers")
  expect_error(instance_sizes(array(0L, dim = c(4, 4, 4)), spacing = c(1, 1)),
               "must be 3 positive numbers")
  expect_error(instance_sizes(two_objects(), spacing = c(0, 1)), "positive")
})

test_that("a stack is refused and says where to apply this instead", {
  expect_error(instance_sizes(array(0L, dim = c(2, 4, 4, 4))), "one map at a time")
})
