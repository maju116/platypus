fit_on_patients <- function(..., envir = parent.frame()) {
  # The dataset has to outlive this helper: evaluate_cases() reads the images again after
  # training, so tying the directory to this frame would delete it in between.
  # Defined in helper-engine.R, which testthat sources at run time and lintr cannot see.
  root <- tiny_patient_dataset(  # nolint: object_usage_linter.
    patients = 6, slices = 2, envir = envir
  )
  spec <- platypus_spec(
    data = segmentation_data(root, root, colormap = binary_colormap),
    models = list(u_net("tiny", input_shape = c(32, 32), blocks = 2, filters = 4,
                        metrics = list(metric_dice(include_background = FALSE)),
                        epochs = 1, batch_size = 2, ...))
  )
  platypus_fit(spec, num_workers = 0)
}

test_that("scores come back one row per case", {
  skip_if_no_splits()
  cases <- evaluate_cases(fit_on_patients())

  expect_s3_class(cases, "platypus_cases")
  expect_equal(nrow(cases), 12)
  expect_true("case" %in% names(cases))
  expect_true(all(cases$dice >= 0 & cases$dice <= 1))
})

test_that("cases are named after the sample, so a bad score can be chased", {
  # A table of scores against 'row 7' is useless: the first thing anyone does with a low
  # score is go and look at the image.
  skip_if_no_splits()
  cases <- evaluate_cases(fit_on_patients())
  expect_match(cases$case[[1]], "^patient\\d+_slice\\d+$")
})

test_that("grouping reports patients instead of slices", {
  skip_if_no_splits()
  cases <- evaluate_cases(fit_on_patients(), group_by = "^(patient\\d+)_")

  expect_equal(nrow(cases), 6)
  expect_true("group" %in% names(cases))
  expect_false("case" %in% names(cases))
})

test_that("summary() gives the distribution and names the worst case", {
  skip_if_no_splits()
  summary_table <- summary(evaluate_cases(fit_on_patients()))

  expect_true(all(c("metric", "n", "mean", "sd", "median", "min", "max") %in%
                    names(summary_table)))
  expect_equal(summary_table$n[[1]], 12)
  expect_true(summary_table$min[[1]] <= summary_table$mean[[1]])
  expect_true(summary_table$mean[[1]] <= summary_table$max[[1]])
  # The column that makes the table actionable rather than decorative.
  expect_true(any(grepl("^worst_", names(summary_table))))
})

test_that("the summary is the engine's, not a second implementation in R", {
  # Means and quantiles are easy enough to write twice, which is the problem: two tables
  # that drift apart are worse than one table. This pins that R is asking the engine.
  skip_if_no_splits()
  cases <- evaluate_cases(fit_on_patients())
  mine <- summary(cases)
  theirs <- platypus:::rows_to_frame(platypus:::shim()$case_summary(
    lapply(seq_len(nrow(cases)), function(i) as.list(cases[i, , drop = FALSE]))
  )$table)
  expect_equal(mine$mean, theirs$mean)
})

test_that("a tiled model is scored on whole images, not on tiles", {
  # Four tiles per image, but still one row per image - and the score is the image's, since
  # the engine accumulates overlaps rather than averaging tile scores.
  skip_if_no_splits()
  cases <- evaluate_cases(fit_on_patients(splits = c(2, 2)))
  expect_equal(nrow(cases), 12)
})

test_that("asking for a model that was not trained lists the ones that were", {
  skip_if_no_splits()
  expect_error(evaluate_cases(fit_on_patients(), model = "nope"), "no model called")
})

test_that("printing shows the first rows and says how to get the distribution", {
  skip_if_no_splits()
  expect_output(print(evaluate_cases(fit_on_patients())), "scores by case")
  expect_output(print(evaluate_cases(fit_on_patients())), "summary\\(\\)")
})
