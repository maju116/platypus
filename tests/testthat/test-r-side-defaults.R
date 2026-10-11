# R sends what it was given, and nothing else (#148).
#
# The rule is §3D.4: defaults live in the engine, because a default sent from R is
# indistinguishable from a value the user chose - which silences a schema, or a weights
# file, that knows better. `JOURNAL.md` §4aa removed R's defaults for that reason and
# missed six: `mode`, `subdirs`, `column_sep` and `shuffle` on `segmentation_data()`, plus
# `annotation_format` and `strict_labels` on `detection_data()`.
#
# Found by `save_spec()`, which writes only what was set, so the first file written from R
# showed exactly what R adds on its own.
#
# Most of this file needs no engine: what a constructor puts in a list is R's own business,
# and that is where the defect lived.

test_that("segmentation_data sends only what it was given", {
  data <- segmentation_data("train/", "valid/", colormap = binary_colormap)
  for (field in c("mode", "subdirs", "column_sep", "shuffle")) {
    expect_null(data[[field]], info = field)
  }
})

test_that("detection_data sends only what it was given", {
  data <- detection_data("train/", "valid/", classes = "cell")
  for (field in c("mode", "subdirs", "column_sep", "shuffle",
                  "annotation_format", "strict_labels")) {
    expect_null(data[[field]], info = field)
  }
})

test_that("a value the caller names survives, including the falsy ones", {
  # The half that makes the test above worth having. Dropping unset fields by deleting the
  # arguments would pass it and break every caller who uses them - and `shuffle = FALSE`
  # and `strict_labels = FALSE` are the two that a naive `if (x)` would silently drop.
  segmentation <- segmentation_data(
    "train/", "valid/", colormap = binary_colormap,
    mode = "config_file", subdirs = c("pics", "msks"), column_sep = ",", shuffle = FALSE
  )
  expect_identical(segmentation$mode, "config_file")
  expect_identical(segmentation$subdirs, c("pics", "msks"))
  expect_identical(segmentation$column_sep, ",")
  expect_identical(segmentation$shuffle, FALSE)

  detection <- detection_data(
    "train/", "valid/", classes = "cell",
    annotation_format = "labelme", strict_labels = FALSE,
    subdirs = c("pics", "boxes"), column_sep = "|", shuffle = FALSE
  )
  expect_identical(detection$annotation_format, "labelme")
  expect_identical(detection$strict_labels, FALSE)
  expect_identical(detection$subdirs, c("pics", "boxes"))
  expect_identical(detection$column_sep, "|")
  expect_identical(detection$shuffle, FALSE)
})

test_that("mode is still sent when this package chooses it", {
  # The one case where R deciding is as good as the user deciding: a `platypus_split`
  # carries three paths, so the data is a config file whether or not anybody said so. If
  # this stopped being sent, the engine would read `nested_dirs` and find no samples.
  split <- structure(
    list(train_path = "t.csv", validation_path = "v.csv", test_path = NULL),
    class = "platypus_split"
  )
  data <- segmentation_data(split, colormap = binary_colormap)
  expect_identical(data$mode, "config_file")
})

test_that("the engine's defaults are the ones R used to send", {
  # Why dropping them changes nothing today, and why it matters anyway: measured here
  # against the schema rather than assumed, so the day the engine moves one of its own
  # defaults this test says that R is no longer in the way of it.
  skip_if_no_engine()
  properties <- platypus:::engine()$spec_schema()$`$defs`$SegmentationData$properties
  expect_identical(properties$mode$default, "nested_dirs")
  expect_identical(unlist(properties$subdirs$default), c("images", "masks"))
  expect_identical(properties$column_sep$default, ";")
  expect_identical(properties$shuffle$default, TRUE)

  detection <- platypus:::engine()$spec_schema()$`$defs`$DetectionData$properties
  expect_identical(detection$annotation_format$default, "pascal_voc")
  expect_identical(detection$strict_labels$default, TRUE)
})

test_that("a run built without them still resolves to the same specification", {
  # The behavioural claim, end to end: the engine fills in what R stopped sending, and the
  # resolved specification is identical to one that names all four. If this failed, the
  # change would be a silent behaviour change rather than a cleanup.
  skip_if_no_engine()
  bare <- platypus_spec(
    data = segmentation_data("t/", "v/", colormap = binary_colormap),
    models = list(u_net("m", input_shape = c(32, 32))),
    check_paths = FALSE
  )
  spelled_out <- platypus_spec(
    data = segmentation_data("t/", "v/", colormap = binary_colormap,
                             mode = "nested_dirs", subdirs = c("images", "masks"),
                             column_sep = ";", shuffle = TRUE),
    models = list(u_net("m", input_shape = c(32, 32))),
    check_paths = FALSE
  )
  expect_identical(as.list(bare), as.list(spelled_out))
})
