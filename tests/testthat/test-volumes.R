# Volumes from R. The tests that matter are the ones about geometry: a mask written without
# the scan's affine is a file that opens and lies.

test_that("classes are named one way or the other, and the check costs no Python", {
  expect_error(segmentation_data("a", "b"), "exactly one of")
  expect_error(
    segmentation_data("a", "b", colormap = binary_colormap, labels = c(0, 1)),
    "exactly one of"
  )
  data <- segmentation_data("a", "b", labels = c(0, 1, 2))
  expect_identical(data$labels, c(0L, 1L, 2L))
  expect_null(data$colormap)
})

test_that("`window` is the name, and `dicom_window` still answers", {
  expect_identical(segmentation_data("a", "b", labels = c(0, 1), window = "lung")$window,
                   "lung")
  expect_identical(
    segmentation_data("a", "b", labels = c(0, 1), dicom_window = "bone")$window,
    "bone"
  )
  expect_error(
    segmentation_data("a", "b", labels = c(0, 1), window = "lung", dicom_window = "bone"),
    "former name"
  )
})

test_that("an unset window is left out of the request entirely", {
  # Same reason as when DICOM arrived: sending a default nobody asked for would break every
  # run against an engine that does not know the field.
  data <- segmentation_data("a", "b", labels = c(0, 1))
  expect_false("window" %in% names(platypus:::compact(data)))
})

test_that("mask_volume turns voxels into millilitres", {
  # 32 voxels of 2 x 2 x 5 mm = 640 mm^3 = 0.64 ml. Checked by hand because this is the
  # number that would go in a report, and an arithmetic slip here is invisible.
  mask <- array(1L, dim = c(10, 10, 4))
  mask[3:6, 3:6, 2:3] <- 2L
  expect_equal(mask_volume(mask, spacing = c(2, 2, 5)), 0.64)
  expect_equal(mask_volume(mask, spacing = c(1, 1, 1)), 0.032)
})

test_that("mask_volume refuses what it cannot measure", {
  expect_error(mask_volume(array(1L, c(4, 4, 2)), spacing = c(1, 1)), "three positive")
  expect_error(mask_volume(array(1L, c(4, 4, 2)), spacing = c(1, 1, -1)), "three positive")
  expect_error(mask_volume(array(1L, c(4, 4)), spacing = c(1, 1, 1)), "three dimensions")
})

test_that("save_masks refuses the volume case a shape can prove", {
  # Five dimensions can only be volumes. Four cannot be told apart from a stack of images
  # with a class axis, and this function does not guess - the documentation points at
  # save_volumes() instead of a check pretending to know.
  expect_error(save_masks(array(1L, c(2, 8, 8, 4, 2)), tempdir(), binary_colormap),
               "save_volumes")
})

test_that("save_volumes states what each rank means rather than guessing", {
  # Four dimensions is a stack of class-index volumes, always. Reading it as one volume's
  # probabilities when the stack happens to be small is how a silent misinterpretation gets
  # into a package - it would write one wrong file and look like it worked.
  skip_if_no_volumes()
  root <- tiny_volume_dataset(cases = 2)
  scans <- sort(list.files(root, pattern = "ct[.]nii[.]gz$", recursive = TRUE,
                           full.names = TRUE))
  written <- save_volumes(array(1L, c(2, 8, 8, 4)), withr::local_tempdir(),
                          reference = scans)
  expect_length(written, 2)

  expect_error(save_volumes(array(1L, c(8, 8)), withr::local_tempdir(), reference = scans),
               "expected 3")
})

test_that("plot_masks insists on a slice for volumes, and refuses one for images", {
  volumes <- array(0, dim = c(2, 8, 8, 4, 3))
  expect_error(plot_masks(volumes), "Choose a slice")
  images <- array(0, dim = c(2, 8, 8, 3))
  expect_error(plot_masks(images, slice = 2), "only applies to volumes")
})

test_that("volume_info reports spacing and voxel size", {
  skip_if_no_volumes()
  root <- tiny_volume_dataset(cases = 2, spacing = c(2, 2, 5))
  scans <- list.files(root, pattern = "ct[.]nii[.]gz$", recursive = TRUE,
                      full.names = TRUE)

  info <- volume_info(scans)
  expect_equal(nrow(info), 2)
  expect_equal(info$spacing_3, c(5, 5))
  # 2 x 2 x 5 mm = 20 mm^3 = 0.02 ml per voxel.
  expect_equal(info$voxel_ml, c(0.02, 0.02))
  expect_equal(info$shape_1, c(8, 8))
})

test_that("a volume spec trains from R", {
  skip_if_no_volumes()
  root <- tiny_volume_dataset()
  spec <- platypus_spec(
    data = segmentation_data(root, root, labels = c(0, 1), window = "soft_tissue"),
    models = list(u_net("unet3d", input_shape = c(8, 8, 4), channels = 1, blocks = 2,
                        filters = 4, epochs = 1, batch_size = 1,
                        metrics = list(metric_dice())))
  )
  fit <- platypus_fit(spec, num_workers = 0)

  expect_s3_class(fit, "platypus_fit")
  masks <- predict(fit, split = "validation")
  expect_equal(dim(masks), c(4L, 8L, 8L, 4L))
})

test_that("a predicted volume is written with the geometry of the scan it came from", {
  # The reason save_volumes() exists. A mask whose affine does not match the scan lands in
  # the wrong place in every viewer, and nothing about the file looks wrong.
  skip_if_no_volumes()
  root <- tiny_volume_dataset(cases = 2, spacing = c(2, 2, 5))
  scans <- sort(list.files(root, pattern = "ct[.]nii[.]gz$", recursive = TRUE,
                           full.names = TRUE))
  out <- withr::local_tempdir()

  masks <- array(1L, dim = c(2, 8, 8, 4))
  masks[, 3:6, 3:6, 2:3] <- 2L

  written <- save_volumes(masks, out, reference = scans)
  expect_length(written, 2)
  expect_true(all(file.exists(written)))
  expect_match(basename(written[[1]]), "^ct[.]nii[.]gz$")

  # Same spacing as the scan, which is the part that makes it overlay.
  expect_equal(volume_info(written)$spacing_3, volume_info(scans)$spacing_3)
  expect_equal(volume_info(written)$shape_1, volume_info(scans)$shape_1)
})

test_that("a mask that does not match its scan is refused rather than written", {
  skip_if_no_volumes()
  root <- tiny_volume_dataset(cases = 1)
  scan <- list.files(root, pattern = "ct[.]nii[.]gz$", recursive = TRUE,
                     full.names = TRUE)
  wrong_size <- array(1L, dim = c(1, 4, 4, 2))
  expect_error(save_volumes(wrong_size, withr::local_tempdir(), reference = scan),
               "does not match")
})

test_that("masks and reference volumes have to line up", {
  skip_if_no_volumes()
  masks <- array(1L, dim = c(2, 8, 8, 4))
  expect_error(save_volumes(masks, withr::local_tempdir(), reference = "only-one.nii.gz"),
               "have to line up")
})

test_that("predicted volumes can be measured in millilitres end to end", {
  # The whole point of carrying spacing: from a prediction to a number that goes in a report
  # without the user doing arithmetic on voxel counts.
  skip_if_no_volumes()
  root <- tiny_volume_dataset(cases = 1, spacing = c(2, 2, 5))
  scan <- list.files(root, pattern = "ct[.]nii[.]gz$", recursive = TRUE, full.names = TRUE)
  info <- volume_info(scan)

  truth <- array(1L, dim = c(8, 8, 4))
  truth[3:6, 3:6, 2:3] <- 2L
  millilitres <- mask_volume(truth, spacing = c(info$spacing_1, info$spacing_2,
                                                info$spacing_3))
  expect_equal(millilitres, 32 * 2 * 2 * 5 / 1000)
})
