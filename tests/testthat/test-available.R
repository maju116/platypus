# The six listings, held to the engine rather than to a list written here.
#
# The whole value of `available_losses()` and its family is that nothing in this package
# maintains them: they are read from the engine's own schema, so a loss added there appears
# in R with nothing edited. A test comparing them against names written in this file would
# assert the opposite - that somebody remembered - which is the failure mode the functions
# exist to remove. So every comparison below asks the engine the same question twice: once
# through R, once through reticulate, and checks the answers are the same set.

listing_from_engine <- function(kind, task = NULL) {
  rows <- if (is.null(task)) {
    reticulate::py_eval(sprintf("__import__('pyplatypus').available_%s()", kind))
  } else {
    reticulate::py_eval(sprintf("__import__('pyplatypus').available_%s(%s)", kind,
                                shQuote(task, type = "cmd")))
  }
  vapply(rows, function(row) row$name, character(1))
}

test_that("every listing names exactly what the engine names", {
  skip_if_no_engine()
  cases <- list(
    list(f = available_losses, kind = "losses", task = "semantic_segmentation"),
    list(f = available_metrics, kind = "metrics", task = "semantic_segmentation"),
    list(f = available_models, kind = "models", task = "semantic_segmentation"),
    list(f = available_models, kind = "models", task = "object_detection"),
    list(f = available_optimizers, kind = "optimizers", task = NULL),
    list(f = available_callbacks, kind = "callbacks", task = NULL),
    list(f = available_tasks, kind = "tasks", task = NULL)
  )
  for (case in cases) {
    got <- if (is.null(case$task)) case$f() else case$f(case$task)
    expect_setequal(got$name, listing_from_engine(case$kind, case$task))
    expect_gt(nrow(got), 0)
    expect_identical(names(got), c("name", "arguments", "description"))
    expect_true(all(nzchar(got$description)))
  }
})

test_that("the counts are what the engine's own unions hold", {
  # A floor each, so the comparison above cannot pass by matching two empty answers - §4's
  # "two numbers that were both zero". These are counts of what the pinned engine accepts,
  # not of what it will accept forever, which is why they are asserted here and written in
  # no help page.
  skip_if_no_engine()
  expect_identical(nrow(available_losses()), 10L)
  expect_identical(nrow(available_metrics()), 4L)
  expect_identical(nrow(available_optimizers()), 8L)
  expect_identical(nrow(available_callbacks()), 7L)
  expect_identical(nrow(available_models()), 4L)
  expect_identical(nrow(available_models("object_detection")), 1L)
  expect_identical(nrow(available_tasks()), 2L)
})

test_that("arguments are a readable column and recover their parts", {
  skip_if_no_engine()
  tversky <- available_losses()[available_losses()$name == "tversky", ]
  expect_identical(strsplit(tversky$arguments, ", ")[[1]], c("alpha", "smooth"))

  # A callback with no arguments at all is the case that makes an empty string meaningful
  # rather than a missing value: `terminate_on_nan` takes nothing, and NA would read as
  # "unknown" instead of "none".
  nan <- available_callbacks()[available_callbacks()$name == "terminate_on_nan", ]
  expect_identical(nan$arguments, "")
  expect_false(is.na(nan$arguments))
})

test_that("detection says what it has instead of a loss to choose", {
  skip_if_no_engine()
  expect_error(available_losses("object_detection"), class = "platypus_error")
  expect_error(available_losses("object_detection"), "box_loss")
  # The two values come from the engine's schema rather than from its prose, so this holds
  # the message to what the specification actually accepts.
  expect_error(available_losses("object_detection"), "offsets")
  expect_error(available_losses("object_detection"), "giou")
})

test_that("detection names the metrics it reports rather than returning none", {
  skip_if_no_engine()
  expect_error(available_metrics("object_detection"), class = "platypus_error")
  for (reported in c("map_50", "map_50_95", "mean_matched_iou")) {
    expect_error(available_metrics("object_detection"), reported, fixed = TRUE)
  }
})

test_that("an unknown task lists the ones there are", {
  skip_if_no_engine()
  expect_error(available_losses("instance_segmentation"), class = "platypus_error")
  expect_error(available_losses("instance_segmentation"), "semantic_segmentation")
  expect_error(available_losses("instance_segmentation"), "object_detection")
})

test_that("the family is spelled one way across both packages", {
  # §3A: one concept, one name. Seven exported functions answering "what can this do",
  # and `show_available_*` for any of them would be the divergence the rule exists to
  # prevent. Checked against the namespace rather than by reading the file.
  exported <- getNamespaceExports("platypus")
  family <- sort(grep("^available_", exported, value = TRUE))
  expect_identical(
    family,
    c("available_callbacks", "available_losses", "available_metrics", "available_models",
      "available_optimizers", "available_tasks", "available_transforms",
      "available_weights")
  )
  expect_identical(length(grep("^show_available", exported, value = TRUE)), 0L)
})
