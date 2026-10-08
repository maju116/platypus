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
  # The Overview is a page rather than a group, so it carries `file` and no `section` - and
  # taking `section` off every entry is an error rather than a mismatch, which is how this
  # first reported the Overview being added. pyplatypus's equivalent filters the same way.
  nested <- Filter(function(entry) !is.null(entry$section), reference$contents)

  expect_identical(
    vapply(nested, `[[`, character(1), "section"),
    vapply(groups, `[[`, character(1), "title"),
    info = "run tools/build-reference.R - the sidebar is stale"
  )

  from_groups <- unlist(lapply(groups, function(g) paste0("man/", unlist(g$contents), ".qmd")))
  from_sidebar <- unlist(lapply(nested, `[[`, "contents"))
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

# The reference groups, in the order both sites show them. Each package carries the subset it
# has: "Setting up" is the reticulate bridge and exists only here, "Looking at the results" is
# plotting which the engine deliberately does not carry, "Masks and volumes" has no public
# counterpart there - measured, the engine's `__all__` holds 24 names and not one is a mask,
# volume or colormap utility - and "Records" and "When something is wrong" are the engine's,
# because this package has no record reader and documents no condition classes.
#
# **Asserting the two lists are equal would assert something false**, which CLAUDE.md did for
# a while: "the same eight groups" was written down and five of eight titles matched. A shared
# order with each side showing its own subset is what is actually true, and it still catches
# the drift that mattered - `export_weights` and `available_weights` were under "Training"
# here while the engine had a "Weights" group, so the same two functions were filed in the one
# place a reader of both sites would not look.
#
# Also in pyplatypus's tests/test_reference_groups.py.
group_order <- c(
  "Setting up",
  "A specification",
  "What a model is made of",
  "The data on disk",
  "Training",
  "Predicting and scoring",
  "Weights",
  "Records",
  "Looking at the results",
  "Masks and volumes",
  "When something is wrong"
)

test_that("the groups are a subsequence of the order pyplatypus shares", {
  groups <- reference_groups()
  skip_if(is.null(groups), "tools/reference-groups.yml not reachable from here")

  titles <- vapply(groups, `[[`, character(1), "title")

  unknown <- setdiff(titles, group_order)
  expect_identical(
    unknown, character(0),
    info = paste(
      "group titles outside the shared order. Adding one means adding it to group_order",
      "here and in pyplatypus, or naming it what the other side already calls it"
    )
  )

  expect_identical(
    titles, group_order[group_order %in% titles],
    info = "the groups are out of the order both sites share"
  )
})

# Where the names that exist on both sides must be filed. The subsequence test above cannot do
# this: deleting a group leaves a shorter subsequence, which is still a subsequence, so it
# passes - verified by mutation. That is exactly the drift this pins against, because it is the
# drift that happened: `export_weights` and `available_weights` were here under "Training"
# while the engine had them under "Weights".
#
# Only names that exist in both packages are listed, so this is a claim about agreement and not
# a second copy of the grouping. Also in pyplatypus's tests/test_reference_groups.py.
shared_placement <- c(
  available_transforms = "What a model is made of",
  split_dataset = "The data on disk",
  export_weights = "Weights",
  available_weights = "Weights",
  # Shared from pyplatypus 0.8.0a1, when this package stopped restating the engine's
  # drawing decisions and started mirroring them.
  drawing_style = "Looking at the results"
)

test_that("the shared names are filed where pyplatypus files them", {
  groups <- reference_groups()
  skip_if(is.null(groups), "tools/reference-groups.yml not reachable from here")

  located <- unlist(lapply(groups, function(g) {
    stats::setNames(rep(g$title, length(g$contents)), unlist(g$contents))
  }))

  for (name in names(shared_placement)) {
    expect_identical(
      unname(located[name]), unname(shared_placement[name]),
      info = sprintf(
        "%s is under %s here and under %s in pyplatypus", name,
        if (is.na(located[name])) "nothing" else located[name], shared_placement[name]
      )
    )
  }
})

test_that("every group says what it is for, because the Overview page publishes it", {
  # `desc` was in this file from the start and rendered nowhere, so three groups had none and
  # nothing noticed. tools/build-reference.R now writes _quarto/overview.qmd from these, which
  # makes an empty one a blank section on a published page rather than a blank field in a
  # configuration nobody reads. The page's topic links are covered by the two tests above,
  # which already check the grouping against the exports in both directions.
  groups <- reference_groups()
  skip_if(is.null(groups), "tools/reference-groups.yml not reachable from here")

  without <- vapply(groups, function(g) is.null(g$desc) || !nzchar(trimws(g$desc)), logical(1))
  expect_identical(
    vapply(groups, `[[`, character(1), "title")[without], character(0),
    info = "a group with no `desc` is a blank section on the Overview page"
  )
})
