# The look of both sites, and the only thing that holds the two copies together.
#
# `_brand.yml` is meant to be identical in this repository and in `maju116/pyplatypus`, and
# its own first comment claimed a test in each repository said so. Nothing read it: renaming
# this package on 2026-10-14 edited the R copy's links, the two diverged, and not one test
# failed in either suite. `#160`.
#
# The receipt lives on this side alone, for the reason the field-prose ledger does. The engine
# owns the look - `pyplatypus/style.py` says "So these are the engine's" outright - so R's copy
# is the copy, and a failure here is the right direction: this package is downstream of the
# engine everywhere else, and a brand change in the engine turning this red until R follows is
# the pin's ratchet in another place.
#
# Why not a digest on each side, which is what the field prose uses: it does not catch
# divergence. Edit the file here, the recorded digest here stops matching, update it, and the
# engine's copy keeps passing against its own older digest - two green suites, two different
# files. And a digest computed on both sides is not even comparable: R's `rlang::hash()`
# hashes a serialized R object rather than bytes.

# One property of the endpoint worth knowing before reading a failure: `raw.githubusercontent`
# serves from a cache, so for a few minutes after a brand change merges in the engine this
# test is red against content that is already correct on `main`. That is the ratchet arriving
# late rather than a divergence, and the failure message - which names the first differing
# line - is what tells the two apart.
engine_brand_url <- "https://raw.githubusercontent.com/maju116/pyplatypus/main/_brand.yml"
engine_control_url <- "https://raw.githubusercontent.com/maju116/pyplatypus/main/README.md"

fetch_lines <- function(address) {
  tryCatch(
    {
      connection <- url(address, open = "rb")
      on.exit(close(connection), add = TRUE)
      readLines(connection, warn = FALSE)
    },
    error = function(e) NULL,
    warning = function(w) NULL
  )
}

test_that("`_brand.yml` is the engine's, line for line", {
  # `.Rbuildignore` keeps this file out of the build, so under `R CMD check` it is genuinely
  # absent. Locating it first means the network is never touched there - a test that reaches
  # the internet under a CRAN check is a different problem.
  ours_path <- NULL
  for (root in c(".", "..", file.path("..", ".."))) {
    candidate <- file.path(root, "_brand.yml")
    if (file.exists(candidate)) {
      ours_path <- candidate
      break
    }
  }
  skip_if(
    is.null(ours_path),
    "_brand.yml is not in the build - run the suite from the repository to compare it"
  )

  theirs <- fetch_lines(engine_brand_url)

  # The positive control, because "no network" and "the file is gone from the engine" are the
  # same silence otherwise. If the engine's README is reachable and its brand file is not,
  # that is a finding rather than a reason to skip.
  if (is.null(theirs) || !length(theirs)) {
    reachable <- length(fetch_lines(engine_control_url)) > 0
    if (reachable) {
      fail(paste0(
        "the engine's README is reachable but ", engine_brand_url, " is not. ",
        "The shared brand file has moved or been renamed in maju116/pyplatypus."
      ))
      return(invisible())
    }
    skip("no network: the engine's copy of _brand.yml could not be fetched")
  }

  # A 404 body, a rate-limit notice or any other wrong page must not read as agreement. Keys
  # rather than a length, because the case that slips through is a wrong body of the right
  # size.
  expect_true(
    any(grepl("^meta:", theirs)) && any(grepl("^color:", theirs)),
    info = "what was fetched is not a brand file; the comparison below would be meaningless"
  )

  ours <- readLines(ours_path, warn = FALSE)

  # Line for line rather than byte for byte: `readLines()` drops the terminators, so a CRLF
  # checkout on Windows cannot produce a false failure. There is no `.gitattributes` in either
  # repository, so that is not hypothetical.
  if (length(ours) != length(theirs)) {
    fail(sprintf(
      "_brand.yml has %d lines here and %d in the engine", length(ours), length(theirs)
    ))
    return(invisible())
  }
  differing <- which(ours != theirs)
  expect_identical(
    ours, theirs,
    info = if (length(differing)) {
      sprintf(
        "%d line(s) differ; the first is %d:\n  here  : %s\n  engine: %s",
        length(differing), differing[1], ours[differing[1]], theirs[differing[1]]
      )
    } else {
      ""
    }
  )
})
