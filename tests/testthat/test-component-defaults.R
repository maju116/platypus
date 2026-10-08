# Every default this package writes down is one the engine also writes down.
#
# Thirteen loss and metric constructors restate the engine's defaults in their own
# signatures - `metric_dice(smooth = 1, include_background = TRUE)` and so on - and when this
# was first measured all thirteen agreed exactly. That was a state and not a guarantee:
# nothing compared them, and nothing would have said when they drifted.
#
# `clDice` is why it stopped being hypothetical. It is the first metric whose
# `include_background` differs from its siblings - FALSE, because the background's skeleton
# lies inside the background by construction and the class scores 1 whatever happened - and a
# default copied from the row above would have silently contradicted the engine.
#
# The models were fixed differently, in `JOURNAL.md` §4aa: their R defaults became NULL and
# the engine supplies them, so a weights file that knows better can be heard. That cannot be
# done here without losing the signature as documentation, so these are compared instead.

test_that("every loss and metric default matches the engine's", {
  skip_if_no_engine()

  schema <- platypus:::engine()$spec_schema()$`$defs`
  theirs <- list()
  for (key in names(schema)) {
    props <- schema[[key]]$properties
    tag <- props$name$const %||% props$name$enum[[1]] %||% NULL
    if (is.null(tag)) next
    family <- if (grepl("Metric$", key)) "metric_" else if (grepl("Loss$", key)) "loss_" else next
    theirs[[paste0(family, tag)]] <- props
  }

  ours <- grep("^(loss|metric)_", getNamespaceExports("platypus"), value = TRUE)
  expect_gt(length(ours), 10)

  for (name in sort(ours)) {
    props <- theirs[[name]]
    expect_false(is.null(props), info = paste(name, "names nothing the engine accepts"))

    for (argument in names(formals(get(name, asNamespace("platypus"))))) {
      ours_default <- formals(get(name, asNamespace("platypus")))[[argument]]
      # R sends nothing for an unset argument, so the engine's default applies and there is
      # nothing to compare - which is the arrangement the model constructors all use.
      if (is.null(ours_default) || identical(ours_default, quote(expr = ))) next
      theirs_default <- props[[argument]]$default
      expect_false(
        is.null(theirs_default),
        info = sprintf("%s(%s=) has a default here and none in the engine", name, argument)
      )
      expect_equal(
        eval(ours_default), theirs_default,
        info = sprintf("%s(%s=)", name, argument)
      )
    }
  }
})

test_that("clDice is the one whose background default differs, on both sides", {
  skip_if_no_engine()

  expect_false(eval(formals(metric_cldice)$include_background))
  for (other in c("metric_dice", "metric_iou", "metric_tversky")) {
    expect_true(eval(formals(get(other))$include_background), info = other)
  }

  theirs <- platypus:::engine()$spec_schema()$`$defs`$ClDiceMetric$properties
  expect_false(theirs$include_background$default)
})
