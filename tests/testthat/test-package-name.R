# The name the tests ask the namespace for has to be a name that resolves.
#
# 2026-10-14, renaming the package to `rplatypus`: five tests called
# `getNamespaceExports("platypus")`, and no test called `skip_if_not_installed`. The call
# raises a `packageNotFoundError`, which **testthat turns into a skip rather than a
# failure** - so the suite reported `FAIL 0` while 78 assertions had quietly stopped
# running, and the skip count was the only symptom. §3D.7 reads that count for this reason;
# this file makes the same mistake fail instead.
#
# It asserts nothing about *which* name is right, on purpose. A test pinning the literal
# would have to be renamed by the same hand that renames everything else, and would be
# wrong in the same direction at the same moment.

test_that("every namespace the tests name is one that resolves", {
  files <- list.files(".", pattern = "\\.R$", full.names = TRUE)
  asked <- unique(unlist(lapply(files, function(f) {
    lines <- readLines(f, warn = FALSE)
    # Comment lines are dropped, and the first run of this test is why: the prose above
    # quotes the call it guards, so the scanner read its own comment and reported a name
    # from 2026-10-14. A probe that can match its own documentation is measuring the file
    # rather than the code.
    lines <- lines[!grepl("^\\s*#", lines)]
    hits <- regmatches(
      lines, regexpr('(?<=getNamespaceExports\\(")[^"]+', lines, perl = TRUE)
    )
    hits[nzchar(hits)]
  })))

  # If this is empty the test is measuring nothing - the calls it guards were renamed,
  # moved or spelled another way, and silence would read as success.
  expect_gt(length(asked), 0L)

  for (name in asked) {
    expect_true(
      requireNamespace(name, quietly = TRUE),
      info = paste0(
        "the tests ask for namespace '", name, "', which is not installed. ",
        "testthat reports that as a skip, not a failure, so the suite stays green while ",
        "the assertions behind it stop running."
      )
    )
  }
})
