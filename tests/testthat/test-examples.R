# Every name an example calls must exist.
#
# The examples that need a trained model are `\dontrun{}` and therefore never execute -
# which is the right call (`\donttest{}` runs under `R CMD check --as-cran` as a separate
# pass, so an example needing the 5 GB engine would fail CRAN's own incoming check, on
# every platform). The cost of `\dontrun{}` is that the code is never seen by anything.
#
# This is what sees it. Not whether the example *works* - that needs the engine - but
# whether it still refers to functions that exist. Documentation rots in the direction of
# the work having been done, and a renamed function or a dropped argument leaves an example
# that reads as current and is not.

rd_examples <- function() {
  # Source `man/` first, installed package second, and the order matters.
  #
  # `tools::Rd_db("platypus")` reads the *installed* package - correct under
  # `R CMD check`, which has just built one, and empty under `pkgload::load_all()`,
  # which is how this package is most often worked on locally. Reading only the
  # installed database left this test skipping in exactly that case: 2 skipped where
  # 0 was expected, and a silently inert tripwire is worse than none. The same hazard
  # made a vignette knit against platypus 0.2.0 and a linter report live helpers as
  # undefined - three times now that the installed package was not the source tree.
  roots <- c(".", "..", file.path("..", ".."))
  for (root in roots) {
    if (dir.exists(file.path(root, "man")) && file.exists(file.path(root, "DESCRIPTION"))) {
      db <- tryCatch(tools::Rd_db(dir = root), error = function(e) list())
      if (length(db)) return(collect(db))
    }
  }
  db <- tryCatch(tools::Rd_db("platypus"), error = function(e) list())
  collect(db)
}

collect <- function(db) {
  out <- list()
  for (name in names(db)) {
    rd <- db[[name]]
    tags <- vapply(rd, function(x) {
      tag <- attr(x, "Rd_tag")
      if (is.null(tag)) "" else tag
    }, character(1))
    blocks <- rd[tags == "\\examples"]
    if (!length(blocks)) next
    code <- paste(unlist(lapply(blocks, as.character)), collapse = "\n")
    out[[name]] <- code
  }
  out
}

called_names <- function(expression) {
  found <- character(0)
  walk <- function(node) {
    if (is.call(node)) {
      head <- node[[1]]
      if (is.name(head)) found <<- c(found, as.character(head))
      for (part in as.list(node)[-1]) if (!missing(part)) walk(part)
    } else if (is.pairlist(node) || is.list(node)) {
      for (part in as.list(node)) walk(part)
    }
  }
  walk(expression)
  found
}

test_that("every Rd example parses", {
  examples <- rd_examples()
  skip_if(length(examples) == 0, "no installed Rd database to read")
  for (name in names(examples)) {
    parsed <- tryCatch(parse(text = examples[[name]]), error = function(e) e)
    expect_false(inherits(parsed, "error"),
                 label = paste0(name, " example does not parse: ",
                                if (inherits(parsed, "error")) conditionMessage(parsed) else ""))
  }
})

test_that("every function an example calls still exists", {
  examples <- rd_examples()
  skip_if(length(examples) == 0, "no installed Rd database to read")

  reachable <- function(name) {
    exists(name, envir = asNamespace("platypus"), inherits = FALSE) ||
      exists(name, envir = baseenv()) ||
      exists(name, envir = globalenv(), inherits = TRUE)
  }

  unknown <- list()
  for (name in names(examples)) {
    parsed <- tryCatch(parse(text = examples[[name]]), error = function(e) NULL)
    if (is.null(parsed)) next
    calls <- unique(unlist(lapply(parsed, called_names)))
    # Names bound inside the example itself are not calls to anything we ship.
    missing_here <- Filter(function(f) !reachable(f), calls)
    if (length(missing_here)) unknown[[name]] <- missing_here
  }

  expect_equal(
    unknown, list(),
    info = paste("examples calling functions that do not exist:",
                 paste(names(unknown), collapse = ", "))
  )
})
