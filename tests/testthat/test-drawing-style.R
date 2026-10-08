# The mirror, held to what it mirrors.
#
# `R/style.R` copies five decisions out of `pyplatypus.style` rather than fetching them,
# because drawing in R must keep working with no Python at all. That copy is only honest if
# something compares it, and this is that something: without it the two agree exactly as
# long as somebody keeps laying their figures side by side, which is how five divergences
# were found on 2026-10-08 and is not a method.

test_that("every drawing decision equals the engine's", {
  skip_if_no_engine()

  result <- platypus:::shim()$drawing_style()
  expect_true(isTRUE(result$ok), info = result$message %||% "")
  theirs <- result$style
  ours <- drawing_style()

  expect_setequal(names(ours), names(theirs))

  # Named vectors in R against dicts in Python: compare by name, so a reordering on either
  # side is not a failure and a renamed key is.
  for (field in c("agreement_colours", "box_colours")) {
    expect_setequal(names(ours[[field]]), names(theirs[[field]]))
    for (key in names(ours[[field]])) {
      expect_identical(
        unname(ours[[field]][[key]]), as.character(theirs[[field]][[key]]),
        info = paste(field, key)
      )
    }
  }

  # An ordered palette, so unlike the two above this one is compared in order: the engine
  # cycles it by position, and a reordering would give a different class a different colour.
  expect_identical(
    unname(ours$class_colours), as.character(unlist(theirs$class_colours))
  )

  expect_equal(ours$overlay_alpha, as.numeric(theirs$overlay_alpha))
  expect_equal(ours$box_min_score, as.numeric(theirs$box_min_score))
})

test_that("the label format means the same thing in both languages", {
  skip_if_no_engine()

  # Not compared as strings: Python writes `{label} {score:.2f}` and R writes `%s %.2f`,
  # which are the same instruction in two notations and would never be `identical()`. What
  # has to match is what they produce, so both are applied to the same box.
  theirs <- platypus:::shim()$drawing_style()$style$box_label_format
  rendered_py <- reticulate::py_eval(
    sprintf("%s.format(label='WBC', score=0.98765)", deparse(theirs))
  )
  rendered_r <- sprintf(drawing_style()$box_label_format, "WBC", 0.98765)

  expect_identical(rendered_r, rendered_py)
  expect_identical(rendered_r, "WBC 0.99")
})

test_that("the drawing functions take their defaults from the mirror and need no engine", {
  # The property that decided the design: someone with masks in R can draw them without a
  # five-gigabyte torch download. Asserted on the formals rather than by running, so it
  # holds whether or not Python is present on the machine running the suite.
  expect_identical(
    deparse(formals(overlay_mask)$alpha), "drawing_style()$overlay_alpha"
  )
  expect_identical(
    deparse(formals(overlay_agreement)$colours), "drawing_style()$agreement_colours"
  )
  expect_identical(
    deparse(formals(plot_boxes)$colours), "drawing_style()$box_colours"
  )
  expect_identical(
    deparse(formals(plot_boxes)$min_score), "drawing_style()$box_min_score"
  )
  expect_identical(
    deparse(formals(plot_masks)$alpha), "drawing_style()$overlay_alpha"
  )
})

test_that("no drawing decision is written down twice", {
  # A literal that crept back into a signature would pass every test above, because the
  # tests compare the mirror and not the code. This reads the sources instead.
  #
  # `R/plot.R` keeps one 0.55 on purpose - the anchor plot's point transparency, a ggplot
  # aesthetic that happens to share a number with `overlay_alpha` - so it is named here
  # rather than permitted everywhere.
  sources <- list.files("../../R", pattern = "[.]R$", full.names = TRUE)
  if (!length(sources)) skip("sources are not next to the tests in an installed package")

  # Code only. These values are discussed in comments in `R/style.R`, and a comment cannot
  # make a second drawing - counting prose would make this fail for being well documented,
  # which is the shape of several checks this project has had to repair.
  code <- unlist(lapply(sources, readLines))
  code <- code[!grepl("^\\s*#", code)]

  for (value in c(unname(drawing_style()$agreement_colours),
                  unname(drawing_style()$box_colours))) {
    expect_length(grep(value, code, value = TRUE, fixed = TRUE), 1L)
  }

  alphas <- grep("0\\.55", code, value = TRUE)
  alphas <- alphas[!grepl("overlay_alpha = 0.55", alphas, fixed = TRUE)]
  expect_true(
    all(grepl("size = size", alphas, fixed = TRUE)),
    info = paste("an unexplained 0.55:", paste(alphas, collapse = " | "))
  )
})
