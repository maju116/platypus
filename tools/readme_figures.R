# The figures in README.md, made by running the package.
#
# They sit in the most-read file there is, so they should not be screenshots of a session
# nobody can find again. Run from the package root:
#
#     PLATYPUS_BCCD=/path/to/BCCD Rscript tools/readme_figures.R boxes
#     PLATYPUS_DSBOWL=/path/to/stage1_train:/path/to/stage1_validation \
#         Rscript tools/readme_figures.R masks
#
# Neither trains. `boxes` needs BCCD's images and the published `bccd-yolo3` weights - the
# point of the figure is that a detector you did not fit puts boxes on a photograph - and
# `masks` needs `dsbowl-unet` and every Data Science Bowl case those weights were split out
# of, which may be in one directory or several, colon-separated.
#
# `masks` insists on all of them because the weights were measured on a seeded 80/20 split
# of the whole set, and its held-out 134 can only be recovered by making that split again.
# A directory named `stage1_validation` is not it: on the machine this was written on, three
# of the five cases the model card names as its worst are in the *training* directory beside
# it, so drawing Dice from there would print numbers for images the model was trained on.
# The check stayed in - this refuses to draw unless the split it reconstructs reproduces the
# card's own mean and worst case.
#
# The README-masks.png this replaces was made by hand in September, before this script
# existed, and had no provenance at all. That was the thing worth not repeating.

which <- commandArgs(trailingOnly = TRUE)
if (!length(which)) which <- "boxes"

pkgload::load_all(".", quiet = TRUE)

figure <- function(name, width, height, expr) {
  path <- file.path("man", "figures", name)
  grDevices::png(path, width = width, height = height, units = "in", res = 110)
  on.exit(grDevices::dev.off(), add = TRUE)
  print(expr)
  message("wrote ", path)
}

if ("boxes" %in% which) {
  root <- Sys.getenv("PLATYPUS_BCCD")
  stopifnot(nzchar(root))

  # BCCD's own test split, which `bccd-yolo3` did not see. The figure is a figure and not a
  # score: the numbers belong in the model card, measured over the whole split.
  names <- trimws(readLines(file.path(root, "ImageSets", "Main", "test.txt"), warn = FALSE))
  names <- names[nzchar(names)]
  frame <- data.frame(
    images = file.path(root, "JPEGImages", paste0(names, ".jpg")),
    annotations = file.path(root, "Annotations", paste0(names, ".xml"))
  )
  csv <- file.path(tempdir(), "bccd-readme.csv")
  utils::write.csv(frame, csv, row.names = FALSE)

  cells <- c("RBC", "WBC", "Platelets")
  fit <- platypus_fit(platypus_spec(
    data = detection_data(csv, csv, classes = cells, mode = "config_file"),
    models = list(yolo3("cells", input_shape = c(416, 416),
                        weights = "bccd-yolo3", fit = FALSE))
  ), verbose = FALSE)

  found <- predict(fit, split = "validation")

  # `plot_boxes()` is a vertical montage, so a README figure is one image rather than a
  # strip - and it should be an image holding all three classes, since a frame of red cells
  # alone would show a third of what the detector does. Chosen by reading the annotations,
  # not by eye.
  wanted <- vapply(frame$annotations, function(path) {
    found_in <- xml2::xml_text(xml2::xml_find_all(xml2::read_xml(path), "//object/name"))
    all(cells %in% found_in)
  }, logical(1))
  stopifnot(any(wanted))
  show <- which(wanted)[1]
  message("drawing ", basename(frame$images[show]), " - the first test image with all three classes")

  images <- read_images(frame$images[show], size = NULL)

  # Printed so the README's caption is transcribed rather than counted off the picture.
  drawn <- found[[show]][found[[show]]$score >= 0.5, ]
  counted <- table(factor(drawn$name, levels = cells))
  message("drawn at min_score 0.5: ",
          paste(names(counted), counted, sep = " ", collapse = ", "))

  figure("README-boxes.png", width = 7.5, height = 4.8,
         plot_boxes(images, found[show], min_score = 0.5, text_size = 2.8,
                    labels = basename(frame$images[show])))
}

if ("masks" %in% which) {
  roots <- strsplit(Sys.getenv("PLATYPUS_DSBOWL"), ":", fixed = TRUE)[[1]]
  roots <- roots[nzchar(roots)]
  stopifnot(length(roots) >= 1)

  # One pool of every case, so the split can be made over the whole set the way the weights
  # were. Symbolic links rather than copies: it is 670 directories of images.
  pool <- file.path(tempdir(), "dsbowl-cases")
  unlink(pool, recursive = TRUE)
  dir.create(pool, recursive = TRUE)
  for (root in roots) {
    for (case in list.dirs(root, recursive = FALSE)) {
      file.symlink(normalizePath(case), file.path(pool, basename(case)))
    }
  }
  message(length(list.dirs(pool, recursive = FALSE)), " cases from ", length(roots), " root(s)")

  split <- split_dataset(pool, file.path(tempdir(), "dsbowl-splits"),
                         fractions = c(0.8, 0.2), seed = 1)

  fit <- platypus_fit(platypus_spec(
    data = segmentation_data(split$train_path, split$validation_path,
                             colormap = binary_colormap, mode = "config_file"),
    # Architecture, blocks and filters are adopted from the weights. Restating them here
    # would let the specification disagree with the file, and the file would win.
    models = list(u_net("nuclei", input_shape = c(256, 256),
                        weights = "dsbowl-unet", fit = FALSE,
                        metrics = list(metric_dice(include_background = FALSE))))
  ), verbose = FALSE)

  cases <- evaluate_cases(fit)
  message(sprintf("  split: n=%d mean=%.4f worst=%.4f",
                  nrow(cases), mean(cases$dice), min(cases$dice)))
  message("  card : n=134 mean=0.9205 worst=0.7240")
  if (nrow(cases) != 134 || abs(mean(cases$dice) - 0.9205) > 5e-4 ||
      abs(min(cases$dice) - 0.7240) > 5e-4) {
    stop("this split does not reproduce the model card's numbers, so these are not the ",
         "images `dsbowl-unet` was held out from. Drawing Dice for them would be leakage ",
         "wearing the costume of a result.", call. = FALSE)
  }

  # The best, two from the middle of the ranking and the worst, rather than four good ones:
  # a README that shows only what a model does well is an advertisement.
  ranked <- order(cases$dice)
  shown <- c(ranked[length(ranked)], ranked[length(ranked) %/% 2], ranked[length(ranked) %/% 3],
             ranked[1])

  listing <- utils::read.csv(split$validation_path, stringsAsFactors = FALSE)
  images <- read_images(listing$images[shown], size = c(256, 256))
  # One file per nucleus, which is the layout this dataset is known for, so the truth for
  # one image is the union of its masks - read at the size the model saw, with `nearest`,
  # which `read_masks` does.
  truth <- vapply(shown, function(i) {
    unite_masks(lapply(
      strsplit(listing$masks[i], ";", fixed = TRUE)[[1]],
      function(path) read_masks(path, binary_colormap, size = c(256, 256))[1, , ]
    ))
  }, matrix(0L, 256, 256))
  truth <- aperm(truth, c(3, 1, 2))

  predictions <- predict(fit, split = "validation")

  figure("README-masks.png", width = 9.5, height = 9.0,
         plot_masks(images, prediction = predictions[shown, , ], truth = truth,
                    colormap = binary_colormap,
                    labels = sprintf("dice %.3f", cases$dice[shown])))
  message("  dice drawn: ", paste(sprintf("%.3f", cases$dice[shown]), collapse = ", "))
}
