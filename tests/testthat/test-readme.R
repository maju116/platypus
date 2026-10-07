# The README's sections, in the order both packages show them.
#
# Same invariant as the reference groups and for the same reason: the two READMEs cannot be
# equal - this one has to tell its users what happened to `yolo3()`, and the engine has no
# predecessor with users - but they can show the same sections in the same order, so a reader
# arriving from the other repository is not starting again.
#
# Checked rather than left to care. Four claims in the engine's README had gone stale while
# nobody read it, including two that were simply false: that the R surface did not exist, and
# that detection was not on PyPI. This file is the most-read page in the repository and the
# one nothing else looks at.

readme_path <- function() {
  for (candidate in c(".", "..", file.path("..", ".."), file.path("..", "..", ".."))) {
    path <- file.path(candidate, "README.md")
    if (file.exists(path)) {
      return(path)
    }
  }
  NULL
}

# Also in pyplatypus's tests/test_readme.py. "About the engine" is this package's alone -
# the engine is not a dependency of itself - and the two `###` subsections about coming from
# 0.1.1 have no counterpart there, because nothing was released before the rewrite.
section_order <- c(
  "Installing",
  "A worked example",
  "The same thing from a file",
  "What is in it",
  "Boxes instead of masks",
  "A backbone instead of the built-in encoder",
  "Learning it",
  "What is not in it yet",
  "About the engine",
  "Licence"
)

readme_sections <- function(path) {
  lines <- readLines(path, warn = FALSE)
  # A `## ` inside a fenced block is not a heading, and this README has R and YAML in it.
  fenced <- cumsum(grepl("^\\s*```", lines)) %% 2 == 1
  fence_line <- grepl("^\\s*```", lines)
  heads <- grepl("^## ", lines) & !fenced & !fence_line
  trimws(sub("^## ", "", lines[heads]))
}

test_that("the README sections are a subsequence of the order pyplatypus shares", {
  path <- readme_path()
  skip_if(is.null(path), "README.md not reachable from here")

  got <- readme_sections(path)
  expect_gt(length(got), 8)

  expect_identical(
    setdiff(got, section_order), character(0),
    info = paste(
      "sections outside the shared order. Adding one means adding it to section_order here",
      "and in pyplatypus, or naming it what the other README calls it"
    )
  )
  expect_identical(
    got, section_order[section_order %in% got],
    info = "the README is out of the order both packages share"
  )
})

test_that("the sections that must be here are", {
  # The subsequence test permits absence, which is how the reference-groups test let drift
  # back in until it was pinned. Every one of these has something to say in both packages.
  path <- readme_path()
  skip_if(is.null(path), "README.md not reachable from here")

  expect_identical(
    setdiff(section_order, readme_sections(path)), character(0),
    info = "a section the shared order requires is missing"
  )
})
