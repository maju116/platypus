context("yolo3_generator")

test_that("yolo3_generator  accepts only correct inputs.", {
  images_path <- ""
  annot_path <- ""
  correct_only_images <- TRUE
  correct_net_h <- 416
  correct_net_w <- 416
  correct_grayscale <- FALSE
  scale <- 1 / 255
  batch_size <- 1
  correct_shuffle <- FALSE
  correct_labels <- coco_labels
  correct_annot_format <- "pascal_voc"
  correct_anchors <- coco_anchors

  incorrect_net_h <- 415
  incorrect_net_w <- -3
  incorrect_grayscale <- "abc"
  incorrect_labels = c()
  incorrect_annot_format = "xml"
  incorrect_only_images <- NA
  incorrect_shuffle <- NA
  incorrect_anchors <- 1:3

  expect_output(yolo3_generator(annot_path, images_path, correct_only_images,
                                correct_net_h, correct_net_w, correct_annot_format,
                                correct_grayscale, scale, correct_anchors,
                                correct_labels, batch_size, correct_shuffle))
  expect_error(yolo3_generator(annot_path, images_path, incorrect_only_images,
                               correct_net_h, correct_net_w, correct_annot_format,
                               correct_grayscale, scale, correct_anchors,
                               correct_labels, batch_size, correct_shuffle))
  expect_error(yolo3_generator(annot_path, images_path, correct_only_images,
                               incorrect_net_h, correct_net_w, correct_annot_format,
                               correct_grayscale, scale, correct_anchors,
                               correct_labels, batch_size, correct_shuffle))
  expect_error(yolo3_generator(annot_path, images_path, correct_only_images,
                               correct_net_h, incorrect_net_w, correct_annot_format,
                               correct_grayscale, scale, correct_anchors,
                               correct_labels, batch_size, correct_shuffle))
  expect_error(yolo3_generator(annot_path, images_path, correct_only_images,
                               correct_net_h, correct_net_w, incorrect_annot_format,
                               correct_grayscale, scale, correct_anchors,
                               correct_labels, batch_size, correct_shuffle))
  expect_error(yolo3_generator(annot_path, images_path, correct_only_images,
                               correct_net_h, correct_net_w, correct_annot_format,
                               incorrect_grayscale, scale, correct_anchors,
                               correct_labels, batch_size, correct_shuffle))
  expect_error(yolo3_generator(annot_path, images_path, correct_only_images,
                               correct_net_h, correct_net_w, correct_annot_format,
                               correct_grayscale, scale, incorrect_anchors,
                               correct_labels, batch_size, correct_shuffle))
  expect_error(yolo3_generator(annot_path, images_path, correct_only_images,
                               correct_net_h, correct_net_w, correct_annot_format,
                               correct_grayscale, scale, correct_anchors,
                               incorrect_labels, batch_size, correct_shuffle))
  expect_error(yolo3_generator(annot_path, images_path, correct_only_images,
                               correct_net_h, correct_net_w, correct_annot_format,
                               correct_grayscale, scale, correct_anchors,
                               correct_labels, batch_size, incorrect_shuffle))
})
