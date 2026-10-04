# The figures in README.md, made by running the package.
#
# They sit in the most-read file there is, so they should not be screenshots of a session
# nobody can find again. Run from the package root:
#
#     PLATYPUS_BCCD=/path/to/BCCD Rscript tools/readme_figures.R boxes
#
# `boxes` needs BCCD's images and the published `bccd-yolo3` weights, and trains nothing -
# the point of the figure is that a detector you did not fit puts boxes on a photograph.
#
# `masks` is listed for completeness and is NOT what produced the README-masks.png now in
# the repo: that one was made by hand in September, before this script existed, and
# regenerating it would change a committed figure to prove nothing. Use it for a new one.

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
  stop("See the note at the top of this file: `masks` would overwrite a committed figure. ",
       "Edit this script deliberately if that is what you want.", call. = FALSE)
}
