# Boxes, from R.
#
# The tests that need no Python come first, because they are the ones that run everywhere
# and they cover the part this package is actually responsible for: saying no clearly, and
# turning what the engine returns into something an R user can use.

test_that("detection_data carries the task so nobody has to type it", {
  data <- detection_data("train/", "valid/", classes = c("RBC", "WBC"))
  expect_s3_class(data, "platypus_detection_data")
  expect_identical(attr(data, "task"), "detection")
  expect_identical(data$classes, c("RBC", "WBC"))
  expect_identical(data$subdirs, c("images", "annotations"))
  expect_identical(data$annotation_format, "pascal_voc")
})

test_that("a segmentation data block has no task attribute and defaults to segmentation", {
  data <- segmentation_data("train/", "valid/", colormap = binary_colormap)
  expect_null(attr(data, "task"))
  expect_identical(platypus:::task_of(data), "segmentation")
})

test_that("classes must be named, distinct and non-empty", {
  expect_error(detection_data("a/", "b/"), "must name the classes")
  expect_error(detection_data("a/", "b/", classes = c("RBC", "RBC")), "distinct")
  expect_error(detection_data("a/", "b/", classes = c("RBC", "")), "non-empty")
})

test_that("one class is allowed, because there is no background entry", {
  expect_identical(detection_data("a/", "b/", classes = "cell")$classes, "cell")
})

test_that("coordinates only mean something for Pascal VOC", {
  expect_error(
    detection_data("a/", "b/", classes = "cell", annotation_format = "labelme",
                   coordinates = "zero_based"),
    "continuous"
  )
  expect_identical(
    detection_data("a/", "b/", classes = "cell", coordinates = "zero_based")$coordinates,
    "zero_based"
  )
})

test_that("a split carries mask paths, so it is refused for detection", {
  fake <- structure(list(), class = "platypus_split")
  expect_error(detection_data(fake, "b/", classes = "cell"), "annotation paths")
})


# --- yolo3 ------------------------------------------------------------------------------

test_that("yolo3 refuses an input the grids cannot tile", {
  expect_error(yolo3("d", input_shape = c(400, 400)), "divisible by 32")
  expect_error(yolo3("d", input_shape = c(416, 400)), "divisible by 32")
  expect_silent(yolo3("d", input_shape = c(416, 416)))
})

test_that("yolo3 is two-dimensional", {
  expect_error(yolo3("d", input_shape = c(64, 64, 32)), "detection is 2D")
})

test_that("yolo3 offers no loss and no metrics", {
  # Not an omission: YOLOv3's objective is part of its architecture and mean average
  # precision is not one option among several, so there is nothing to choose.
  arguments <- names(formals(yolo3))
  expect_false("loss" %in% arguments)
  expect_false("metrics" %in% arguments)
})

test_that("anchors and weights together are refused, because loading adopts the file's", {
  expect_error(
    yolo3("d", input_shape = c(416, 416), weights = "bccd-yolo3",
          anchors = list(list(c(0.3, 0.3)), list(c(0.2, 0.2)), list(c(0.1, 0.1)))),
    "not both"
  )
})

test_that("anchors are fractions, and pixels say so by name", {
  # The mistake someone makes once: COCO publishes its nine in pixels at a 416 input.
  coco <- list(
    list(c(116, 90), c(156, 198), c(373, 326)),
    list(c(30, 61), c(62, 45), c(59, 119)),
    list(c(10, 13), c(16, 30), c(33, 23))
  )
  expect_error(yolo3("d", input_shape = c(416, 416), anchors = coco),
               "dividing by 416")

  fractions <- lapply(coco, function(group) lapply(group, function(p) p / 416))
  model <- yolo3("d", input_shape = c(416, 416), anchors = fractions)
  expect_length(model$anchors, 3L)
  expect_equal(model$anchors[[1]][[1]], c(116, 90) / 416)
})

test_that("anchors accept an array as well as nested lists", {
  values <- array(0, dim = c(3, 2, 2))
  values[, , 1] <- c(0.4, 0.2, 0.1)
  values[, , 2] <- c(0.4, 0.2, 0.1)
  model <- yolo3("d", input_shape = c(416, 416), anchors = values, anchors_per_grid = 2)
  expect_length(model$anchors, 3L)
  expect_length(model$anchors[[1]], 2L)
})

test_that("uneven anchor groups are refused, because the head is one tensor", {
  uneven <- list(list(c(0.5, 0.5), c(0.4, 0.4)), list(c(0.2, 0.2)), list(c(0.1, 0.1)))
  expect_error(yolo3("d", input_shape = c(416, 416), anchors = uneven),
               "same number of anchors")
})

test_that("the wrong number of groups is refused", {
  expect_error(
    yolo3("d", input_shape = c(416, 416), anchors = list(list(c(0.1, 0.1)))),
    "three groups"
  )
})


# --- the task is inferred, and mixing is refused ------------------------------------------

test_that("mixing a detection data block with a segmentation model is refused by name", {
  expect_error(
    platypus:::agreed_task(
      detection_data("a/", "b/", classes = "cell"),
      list(u_net("u", input_shape = c(64, 64)))
    ),
    "mixes them"
  )
})

test_that("the refusal names the model that disagrees", {
  caught <- tryCatch(
    platypus:::agreed_task(
      detection_data("a/", "b/", classes = "cell"),
      list(yolo3("good", input_shape = c(416, 416)),
           u_net("bad", input_shape = c(64, 64)))
    ),
    error = function(e) conditionMessage(e)
  )
  expect_match(caught, "good is detection")
  expect_match(caught, "bad is segmentation")
})

test_that("an agreeing specification reports its one task", {
  expect_identical(
    platypus:::agreed_task(detection_data("a/", "b/", classes = "cell"),
                           list(yolo3("d", input_shape = c(416, 416)))),
    "detection"
  )
  expect_identical(
    platypus:::agreed_task(segmentation_data("a/", "b/", colormap = binary_colormap),
                           list(u_net("u", input_shape = c(64, 64)))),
    "segmentation"
  )
})


# --- plot_boxes, which needs no engine ----------------------------------------------------

test_that("plot_boxes draws boxes on an image stack", {
  skip_if_not_installed("ggplot2")
  images <- array(60, dim = c(2, 64, 96, 3))
  found <- list(
    one = data.frame(xmin = 10, ymin = 12, xmax = 40, ymax = 44, score = 0.9,
                     name = "cell"),
    two = data.frame(xmin = 50, ymin = 20, xmax = 80, ymax = 50, score = 0.8,
                     name = "cell")
  )
  plot <- plot_boxes(images, found)
  expect_s3_class(plot, "ggplot")
})

test_that("plot_boxes flips y, so a box near the top of the image is drawn near the top", {
  skip_if_not_installed("ggplot2")
  # The assertion that would catch a mirrored draw. An image row near y=0 is at the *top*
  # of the picture and near the *maximum* y in ggplot's coordinates, and getting that
  # backwards looks plausible on a symmetric figure.
  images <- array(60, dim = c(1, 100, 100, 3))
  near_top <- data.frame(xmin = 10, ymin = 5, xmax = 30, ymax = 25, score = 1,
                         name = "a")
  rows <- platypus:::box_rows(near_top, "prediction", offset = 0, tile_h = 100,
                              min_score = 0.5)
  expect_equal(rows$ymin, 100 - 25)
  expect_equal(rows$ymax, 100 - 5)
  expect_gt(rows$ymax, 50)
})

test_that("plot_boxes drops boxes below min_score and keeps the rest", {
  skip_if_not_installed("ggplot2")
  frame <- data.frame(xmin = c(1, 2), ymin = c(1, 2), xmax = c(10, 20),
                      ymax = c(10, 20), score = c(0.2, 0.9), name = c("a", "b"))
  kept <- platypus:::box_rows(frame, "prediction", 0, 100, min_score = 0.5)
  expect_equal(nrow(kept), 1L)
  expect_match(kept$label, "^b 0.90$")
})

test_that("plot_boxes refuses a count that cannot line up", {
  skip_if_not_installed("ggplot2")
  images <- array(60, dim = c(3, 64, 64, 3))
  expect_error(plot_boxes(images, list(data.frame(xmin = 1, ymin = 1, xmax = 2,
                                                 ymax = 2))),
               "line up one to one")
})

test_that("plot_boxes needs the four coordinate columns", {
  skip_if_not_installed("ggplot2")
  images <- array(60, dim = c(1, 64, 64, 3))
  expect_error(plot_boxes(images, data.frame(x = 1, y = 2)), "need the columns")
})

test_that("an image with nothing found is still an entry", {
  skip_if_not_installed("ggplot2")
  images <- array(60, dim = c(2, 64, 64, 3))
  empty <- data.frame(xmin = numeric(0), ymin = numeric(0), xmax = numeric(0),
                      ymax = numeric(0), score = numeric(0), name = character(0))
  found <- list(empty, data.frame(xmin = 5, ymin = 5, xmax = 20, ymax = 20,
                                  score = 0.9, name = "a"))
  expect_s3_class(plot_boxes(images, found), "ggplot")
})


# --- with the engine ----------------------------------------------------------------------

test_that("a detector trains, scores and returns boxes in each image's own pixels", {
  skip_if_no_detection()
  skip_if_not_installed("png")

  root <- withr::local_tempdir()
  write_detection_split(file.path(root, "train"), "train", 8, seed = 1)
  write_detection_split(file.path(root, "valid"), "valid", 4, seed = 2)

  spec <- platypus_spec(
    data = detection_data(file.path(root, "train"), file.path(root, "valid"),
                          classes = c("square", "bar")),
    models = list(yolo3("d", input_shape = c(128, 128), anchors_per_grid = 2,
                        epochs = 2, batch_size = 2)),
    seed = 1
  )
  fit <- platypus_fit(spec, device = "cpu")
  expect_identical(fit$task, "detection")

  # The four parts of the loss, for train and validation. A YOLOv3 total means nothing
  # alone - it has a floor above zero that depends on the data.
  expect_true(all(c("train_loss", "train_coordinates", "val_loss") %in%
                    names(fit$history)))

  table <- evaluate(fit, split = "validation")
  expect_true(all(c("map_50", "map_50_95", "mean_matched_iou") %in% names(table)))
  # Not here, and deliberately: averaging precision over classes needs a weighting and
  # every weighting is a different claim.
  expect_false("precision" %in% names(table))

  per_class <- evaluate_classes(fit, split = "validation")
  expect_equal(nrow(per_class), 2L)
  expect_true(all(c("precision", "recall") %in% names(per_class)))

  found <- predict(fit, split = "validation")
  expect_length(found, 4L)
  expect_named(found)
  expect_s3_class(found[[1]], "data.frame")
  expect_true(all(c("xmin", "ymin", "xmax", "ymax", "score", "label", "name") %in%
                    names(found[[1]])))
  # Source pixels, not the network's 128x128 frame: the images are 160 wide.
  if (nrow(found[[1]])) {
    expect_lte(max(found[[1]]$xmax), 160 + 1)
    expect_true(all(found[[1]]$name %in% c("square", "bar")))
    # 1-based, like the class indices `predict` returns for masks.
    expect_true(all(found[[1]]$label >= 1L))
  }
})

test_that("the anchors a run used come back, and say they were fitted", {
  skip_if_no_detection()
  skip_if_not_installed("png")

  root <- withr::local_tempdir()
  write_detection_split(file.path(root, "train"), "train", 8, seed = 3)
  write_detection_split(file.path(root, "valid"), "valid", 4, seed = 4)

  spec <- platypus_spec(
    data = detection_data(file.path(root, "train"), file.path(root, "valid"),
                          classes = c("square", "bar")),
    models = list(yolo3("d", input_shape = c(128, 128), anchors_per_grid = 2,
                        epochs = 1, batch_size = 2)),
    seed = 1
  )
  fit <- platypus_fit(spec, device = "cpu")

  anchors <- detection_anchors(fit)
  expect_true(anchors$fitted)
  expect_length(anchors$anchors, 3L)
  expect_length(anchors$anchors[[1]], 2L)
  expect_gt(anchors$mean_iou, 0.3)
  expect_equal(nrow(anchors$per_anchor), 6L)
})

test_that("predict refuses mask arguments on a detector, and only then", {
  # Both halves. `missing()` on a formal argument stops being true once the argument has
  # been assigned to, and `match.arg` assigns - so the first version of this refusal fired
  # for every caller, including the plain `predict(fit)` above. The test below is what
  # caught it, by making the ordinary call first.
  skip_if_no_detection()
  skip_if_not_installed("png")

  root <- withr::local_tempdir()
  write_detection_split(file.path(root, "train"), "train", 8, seed = 5)
  write_detection_split(file.path(root, "valid"), "valid", 4, seed = 6)

  spec <- platypus_spec(
    data = detection_data(file.path(root, "train"), file.path(root, "valid"),
                          classes = c("square", "bar")),
    models = list(yolo3("d", input_shape = c(128, 128), anchors_per_grid = 2,
                        epochs = 1, batch_size = 2)),
    seed = 1
  )
  fit <- platypus_fit(spec, device = "cpu")
  expect_error(predict(fit, split = "validation", space = "source"), "about masks")
  expect_error(predict(fit, split = "validation", type = "probability"), "about masks")
})

test_that("evaluate_classes refuses a segmentation fit, and says what to use", {
  skip_if_no_engine()
  fake <- structure(list(task = "segmentation", models = "u"), class = "platypus_fit")
  expect_error(evaluate_classes(fake), "evaluate_cases")
})

test_that("the engine refuses a detection specification it cannot read", {
  # Against a released engine this is the pin working: an older one rejects `task` and
  # every field under it. Against a current one the spec builds, which is what the skip
  # above already established.
  skip_if_no_detection()
  spec <- platypus_spec(
    data = detection_data(".", ".", classes = c("a", "b")),
    models = list(yolo3("d", input_shape = c(416, 416))),
    check_paths = FALSE
  )
  expect_s3_class(spec, "platypus_spec")
})


# --- the gap a vignette found ------------------------------------------------------------

test_that("every callback the engine offers has an R constructor", {
  # Written because `cosine_annealing` reached the engine's spec and not this package's, so
  # the recipe that produced the published BCCD weights could not be expressed from R at
  # all. Nothing in either test suite could notice: the engine's tests do not know R exists
  # and the R tests only exercise the constructors that exist. Writing the vignette as a
  # user would run it is what found it - the third time in this project (see the `hub`
  # extra), which is why it is now a test.
  skip_if_no_engine()

  # Asked the way the engine itself resolves a callback: the builders' keys. The
  # discriminated union is awkward to introspect and the builder map is the thing that
  # decides, so it is also the right thing to compare against.
  names_in_engine <- reticulate::py_eval(paste0(
    "sorted(s.model_fields['name'].default for s in ",
    "__import__('pyplatypus.training.callbacks', fromlist=['x'])._BUILDERS)"
  ))
  in_r <- sort(sub("^callback_", "", grep("^callback_", getNamespaceExports("platypus"),
                                          value = TRUE)))

  expect_setequal(names_in_engine, in_r)
})
