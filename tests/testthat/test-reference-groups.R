# The reference index, which nothing else checks any more.
#
# `check_pkgdown()` used to fail the build when an exported function was missing from
# `_pkgdown.yml`, and §4v put it *before* the site build precisely because adding a
# function and forgetting the index is the ordinary way to break the site - two seconds
# against four minutes. pkgdown is gone, so that gate has to exist here instead.
#
# What it asserts: every documented topic appears in exactly one group, and every group
# names only topics that exist. Both directions, because the failures are different - the
# first is a page nobody can reach from the sidebar, the second is a sidebar entry leading
# to a 404, and the second is the one a reader notices.

reference_groups <- function() {
  for (root in c(".", "..", file.path("..", ".."))) {
    path <- file.path(root, "tools", "reference-groups.yml")
    if (file.exists(path)) return(yaml::read_yaml(path)$groups)
  }
  NULL
}

man_topics <- function() {
  for (root in c(".", "..", file.path("..", ".."))) {
    man <- file.path(root, "man")
    if (dir.exists(man)) {
      files <- list.files(man, pattern = "[.]Rd$")
      if (length(files)) return(sub("[.]Rd$", "", files))
    }
  }
  db <- tryCatch(tools::Rd_db("platypus"), error = function(e) list())
  sub("[.]Rd$", "", names(db))
}

test_that("every documented topic is in exactly one reference group", {
  groups <- reference_groups()
  skip_if(is.null(groups), "tools/reference-groups.yml not reachable from here")

  topics <- man_topics()
  expect_gt(length(topics), 40)          # or this passes by looking at nothing

  listed <- unlist(lapply(groups, `[[`, "contents"), use.names = FALSE)

  expect_identical(
    sort(setdiff(topics, listed)), character(0),
    info = "documented but in no group, so unreachable from the sidebar"
  )
  expect_identical(
    sort(setdiff(listed, topics)), character(0),
    info = "in a group but not documented, so the sidebar links to nothing"
  )
  expect_identical(
    listed[duplicated(listed)], character(0),
    info = "in more than one group"
  )
})

test_that("the generated sidebar matches the grouping it was generated from", {
  groups <- reference_groups()
  skip_if(is.null(groups), "tools/reference-groups.yml not reachable from here")

  generated <- NULL
  for (root in c(".", "..", file.path("..", ".."))) {
    path <- file.path(root, "altdoc", "quarto_website.yml")
    if (file.exists(path)) {
      generated <- yaml::read_yaml(path)
      break
    }
  }
  skip_if(is.null(generated), "altdoc/quarto_website.yml not reachable from here")

  sections <- generated$website$sidebar$contents
  reference <- Filter(function(entry) identical(entry$section, "Reference"), sections)[[1]]

  expect_identical(
    vapply(reference$contents, `[[`, character(1), "section"),
    vapply(groups, `[[`, character(1), "title"),
    info = "run tools/build-reference.R - the sidebar is stale"
  )

  from_groups <- unlist(lapply(groups, function(g) paste0("man/", unlist(g$contents), ".qmd")))
  from_sidebar <- unlist(lapply(reference$contents, `[[`, "contents"))
  expect_identical(from_sidebar, from_groups,
                   info = "run tools/build-reference.R - the sidebar is stale")
})

# The About section is four links and was uncovered until 2026-10-07: swapping two of them
# left every assertion above passing, because they only read the Reference section. Two
# things are pinned here, and both have already gone wrong once.
#
#  - The order, which matches pyplatypus's sidebar so that a reader moving between the two
#    sites finds the same four things in the same places. That cannot be asserted from here
#    - this repository cannot see the other one - so the order is written out, and the
#    comment in tools/build-reference.R says where it came from.
#  - The pairing of each entry with a file that exists. altdoc builds the placeholder from
#    the basename of whichever file it found, so NEWS.md is $ALTDOC_NEWS; name a placeholder
#    whose file is absent and altdoc deletes the `file:` line and leaves the `text:` line,
#    which is a sidebar entry that cannot be clicked. That is what happened to Licence and
#    Changelog before this test existed.
test_that("the About section is in the agreed order and every entry has a file", {
  root <- NULL
  for (candidate in c(".", "..", file.path("..", ".."))) {
    if (file.exists(file.path(candidate, "altdoc", "quarto_website.yml"))) {
      root <- candidate
      break
    }
  }
  skip_if(is.null(root), "altdoc/quarto_website.yml not reachable from here")

  generated <- yaml::read_yaml(file.path(root, "altdoc", "quarto_website.yml"))
  sections <- generated$website$sidebar$contents
  about <- Filter(function(entry) identical(entry$section, "About"), sections)[[1]]

  expect_identical(
    vapply(about$contents, `[[`, character(1), "text"),
    c("Changelog", "Code of conduct", "Licence", "Citation"),
    info = "run tools/build-reference.R - and keep the order pyplatypus uses"
  )

  for (entry in about$contents) {
    name <- sub("^[$]ALTDOC_", "", entry$file)
    # altdoc writes the citation page from DESCRIPTION, so it is the one with no file.
    if (identical(name, "CITATION")) next
    expect_true(
      any(file.exists(file.path(root, paste0(name, c(".md", ""))))),
      info = paste0(entry$text, " points at $ALTDOC_", name,
                    ", so altdoc needs ", name, ".md at the package root")
    )
  }
})
