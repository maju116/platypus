# A family page that does not name a member of its family.
#
# `callbacks` documented six callbacks and there were seven; `losses` documented nine and
# there were ten. Both missing members - `callback_swa()` and `loss_boundary()` - have a
# page of their own, because each carries a measurement that deserved the space, and
# neither family page was told about them. So `?callbacks` was a complete answer to a
# question nobody had asked since the sixth callback shipped.
#
# This is §4o's rot - documentation goes stale in the direction of work having been done -
# and it is invisible to `R CMD check`, which has no opinion about what a page omits.
#
# The Rd comes from the source `man/` first and the installed database second, the same
# order and for the same reason as test-examples.R: `tools::Rd_db("platypus")` is correct
# under `R CMD check` and empty under `pkgload::load_all()`.

family_rd <- function(topic) {
  roots <- c(".", "..", file.path("..", ".."))
  for (root in roots) {
    man <- file.path(root, "man", paste0(topic, ".Rd"))
    if (file.exists(man)) return(paste(readLines(man, warn = FALSE), collapse = "\n"))
  }
  db <- tryCatch(tools::Rd_db("platypus"), error = function(e) list())
  key <- paste0(topic, ".Rd")
  if (!key %in% names(db)) return(NA_character_)
  paste(unlist(lapply(db[[key]], as.character)), collapse = "\n")
}

families <- list(
  callbacks  = "callback_",
  losses     = "loss_",
  metrics    = "metric_",
  optimizers = "optimizer_"
)

test_that("every member of a family is named on its family page", {
  exported <- getNamespaceExports("platypus")

  for (topic in names(families)) {
    prefix <- families[[topic]]
    members <- sort(grep(paste0("^", prefix), exported, value = TRUE))

    # The scan must find a family, or it passes by looking at nothing.
    expect_gt(length(members), 2)

    page <- family_rd(topic)
    expect_false(is.na(page), info = paste("no Rd found for", topic))

    missing <- members[!vapply(members, function(m) grepl(m, page, fixed = TRUE), logical(1))]
    expect_identical(
      missing, character(0),
      info = paste0("?", topic, " does not name: ", paste(missing, collapse = ", "),
                    " - add it to the page or link its own page from there")
    )
  }
})
