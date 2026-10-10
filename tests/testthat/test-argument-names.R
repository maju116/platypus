# `model` against the engine's `model_name`, which is the rule's own exception and not a
# divergence - but only while the family holds.
#
# §3A: where the two languages would disagree the engine's name wins, *"unless R already has
# a family of that shape"*. The engine says `predict(model_name=)`; R says `model`, and §8's
# queue read that as a divergence to be fixed by a rename on a released surface.
#
# Measured 2026-10-10 before touching anything: 94 names on R's user surface, 8 of them take
# a model selector, all 8 call it `model`, none calls it `model_name`. So it is the exception
# applied consistently, and the rename would have broken eight released functions in order to
# make R disagree with itself.
#
# What this test is for is that the exception was **verbal**. A ninth function arriving as
# `model_name` would quietly dissolve the family that justifies the other eight, and nothing
# would have said so - the argument for keeping `model` would have evaporated while every
# test stayed green.
#
# No reachability guard here, unlike the Rd-reading tests: this reads the namespace, which is
# present under `load_all()` and under `R CMD check` alike.

model_selectors <- function() {
  ns <- asNamespace("platypus")
  # Exports, plus S3 methods on this package's own classes. Not the S3 methods table, which
  # `pkgload::load_all()` populates lazily - it held 4 of them when this was written, and a
  # count that depends on how the package was loaded is not a measurement.
  surface <- sort(unique(c(
    getNamespaceExports(ns),
    grep("\\.platypus_", ls(ns, all.names = TRUE), value = TRUE)
  )))
  found <- character(0)
  for (name in surface) {
    object <- tryCatch(get(name, envir = ns), error = function(e) NULL)
    if (!is.function(object)) next
    hit <- intersect(names(formals(object)), c("model", "model_name"))
    if (length(hit)) found[[name]] <- hit[[1]]
  }
  found
}

test_that("every function that selects a model calls the argument `model`", {
  selectors <- model_selectors()

  # The family has to exist for §3A's exception to mean anything. If this ever reaches zero,
  # the exception is moot and the engine's name should win by the rule's main clause.
  expect_gt(length(selectors), 1L)

  wrong <- names(selectors)[selectors != "model"]
  expect_identical(
    wrong, character(0),
    info = paste(
      "these take `model_name` where R's family says `model`:", paste(wrong, collapse = ", "),
      "\nEither rename it to `model`, or - if the engine's name should now win - rename the",
      "whole family and say so in JOURNAL.md §4aq, because the exception is the only thing",
      "keeping the other members as they are."
    )
  )
})
