# The ecosystem layer works on ordinary arrays, so almost all of this runs without Python.

square <- function(size = 8) {
  mask <- matrix(1L, size, size)
  inner <- seq(max(1L, size %/% 4L), min(size, size - size %/% 4L))
  mask[inner, inner] <- 2L
  mask
}

test_that("a mask is painted with its colormap", {
  coloured <- classes_to_colours(square(), binary_colormap)
  expect_identical(dim(coloured), c(8L, 8L, 3L))
  expect_identical(coloured[1, 1, ], c(0L, 0L, 0L))
  expect_identical(coloured[4, 4, ], c(255L, 255L, 255L))
})

test_that("a class the colormap does not define is refused, and says so", {
  # Counting from 1 is the convention here, and getting it wrong should not produce a
  # picture that looks plausible.
  expect_error(classes_to_colours(matrix(c(1L, 5L), 1), binary_colormap),
               "colormap defines 2 classes")
  expect_error(classes_to_colours(matrix(c(0L, 1L), 1), binary_colormap),
               "counted from 1")
})

test_that("colours and classes are exact inverses", {
  mask <- square()
  back <- colours_to_classes(classes_to_colours(mask, binary_colormap), binary_colormap)
  expect_identical(back$classes, mask)
  expect_identical(back$coverage, 1)
})

test_that("a colormap that does not describe the data shows up as coverage", {
  # The quietest way to train on nothing: every pixel falls through to background and the
  # run looks normal all the way to a model that learned the empty mask.
  coloured <- classes_to_colours(square(), binary_colormap)
  wrong <- colours_to_classes(coloured, list(c(1L, 2L, 3L), c(4L, 5L, 6L)))
  expect_identical(wrong$coverage, 0)
  expect_true(all(wrong$classes == 1L))
})

test_that("an overlay tints the classes and leaves the background alone", {
  image <- array(100, dim = c(8, 8, 3))
  blended <- overlay_mask(image, square(), binary_colormap, alpha = 0.5)
  expect_identical(dim(blended), c(8L, 8L, 3L))
  expect_identical(blended[1, 1, 1], 100L)       # background, untouched
  expect_gt(blended[4, 4, 1], 100L)              # foreground, lightened
})

test_that("a mask of the wrong size is caught rather than recycled", {
  expect_error(overlay_mask(array(0, c(8, 8, 3)), square(4), binary_colormap),
               "8x8 but mask is 4x4")
})

test_that("agreement separates found, missed and invented", {
  # A single overlap score cannot say which of the three you have, and for most clinical
  # questions they cost different things.
  truth <- square()
  prediction <- matrix(1L, 8, 8)
  prediction[3:6, 5:8] <- 2L                      # shifted right: some of each
  image <- array(0, dim = c(8, 8, 3))
  blended <- overlay_agreement(image, prediction, truth)

  expect_identical(dim(blended), c(8L, 8L, 3L))
  expect_gt(blended[4, 5, 2], 0L)                 # hit, green
  expect_gt(blended[4, 3, 1], 0L)                 # missed, red
  expect_gt(blended[4, 8, 1], 0L)                 # invented, yellow
  expect_identical(blended[1, 1, ], c(0L, 0L, 0L))
})

test_that("greyscale and RGBA images are both accepted", {
  mask <- square()
  expect_identical(dim(overlay_mask(matrix(0.5, 8, 8), mask, binary_colormap)),
                   c(8L, 8L, 3L))
  expect_identical(dim(overlay_mask(array(0.5, c(8, 8, 4)), mask, binary_colormap)),
                   c(8L, 8L, 3L))
})

test_that("images in 0-1 and in 0-255 both work", {
  mask <- square()
  expect_lte(max(overlay_mask(array(1, c(8, 8, 3)), mask, binary_colormap)), 255L)
  expect_lte(max(overlay_mask(array(255, c(8, 8, 3)), mask, binary_colormap)), 255L)
})

test_that("several masks of one image become one", {
  # One file per nucleus is how the Data Science Bowl stores its labels, and it is not
  # unusual.
  a <- matrix(1L, 4, 4)
  a[1:2, ] <- 2L
  b <- matrix(1L, 4, 4)
  b[4, ] <- 2L
  united <- unite_masks(list(a, b))
  expect_identical(united[1, 1], 2L)
  expect_identical(united[4, 1], 2L)
  expect_identical(united[3, 1], 1L)
})

test_that("masks of different sizes are refused rather than silently cropped", {
  expect_error(unite_masks(list(matrix(1L, 4, 4), matrix(1L, 8, 8))), "same size")
  expect_error(unite_masks(list()), "empty")
})

test_that("coverage answers how much of the picture each class is", {
  # Worth knowing before training rather than after: a foreground of one percent is a
  # problem the loss has to be chosen for.
  coverage <- mask_coverage(square(), labels = c("background", "nucleus"))
  expect_identical(coverage$label, c("background", "nucleus"))
  expect_identical(sum(coverage$pixels), 64L)
  expect_equal(sum(coverage$fraction), 1)
  expect_equal(coverage$fraction[[2]], sum(square() == 2L) / 64)
})

test_that("the supplied colormaps line up with their labels", {
  expect_length(binary_colormap, length(binary_labels))
  expect_length(voc_colormap, length(voc_labels))
  expect_identical(voc_labels[[1]], "background")
  expect_identical(voc_colormap[[1]], c(0L, 0L, 0L))
  # Every colour distinct, or two classes could not be told apart on disk.
  expect_length(unique(voc_colormap), length(voc_colormap))
})

test_that("masks are written as files that read back identically", {
  # A prediction is only half useful while it is an array in a session. The round trip is
  # the property that matters: what comes back out has to be what went in.
  skip_if_no_engine()
  out <- withr::local_tempdir()
  masks <- array(1L, dim = c(3, 16, 16))
  masks[, 4:12, 4:12] <- 2L

  paths <- save_masks(masks, out, binary_colormap)
  expect_length(paths, 3L)
  expect_true(all(file.exists(paths)))
  expect_identical(read_masks(paths, binary_colormap), masks)
})

test_that("a suffix keeps two models from overwriting each other", {
  skip_if_no_engine()
  out <- withr::local_tempdir()
  mask <- array(1L, dim = c(1, 8, 8))
  save_masks(mask, out, binary_colormap, suffix = "_unet")
  save_masks(mask, out, binary_colormap, suffix = "_linknet")
  expect_length(list.files(out), 2L)
})

test_that("names can come from the images the masks belong to", {
  skip_if_no_engine()
  out <- withr::local_tempdir()
  mask <- array(1L, dim = c(2, 8, 8))
  paths <- save_masks(mask, out, binary_colormap,
                      names = c("a/scan_01.png", "b/scan_02.tif"), suffix = "_m")
  expect_identical(basename(paths), c("scan_01_m.png", "scan_02_m.png"))
})

test_that("probabilities are reduced to classes rather than refused", {
  skip_if_no_engine()
  out <- withr::local_tempdir()
  probabilities <- array(0, dim = c(2, 8, 8, 2))
  probabilities[, , , 1] <- 0.3
  probabilities[, , , 2] <- 0.7
  paths <- save_masks(probabilities, out, binary_colormap)
  expect_identical(unique(as.vector(read_masks(paths, binary_colormap))), 2L)
})

test_that("a mismatched number of names is caught before anything is written", {
  skip_if_no_engine()
  out <- withr::local_tempdir()
  expect_error(
    save_masks(array(1L, dim = c(3, 8, 8)), out, binary_colormap, names = c("a", "b")),
    "2 entries but there are 3"
  )
  expect_length(list.files(out), 0L)
})

test_that("the output directory is created rather than demanded", {
  skip_if_no_engine()
  out <- file.path(withr::local_tempdir(), "does", "not", "exist")
  paths <- save_masks(array(1L, dim = c(1, 8, 8)), out, binary_colormap)
  expect_true(file.exists(paths))
})
