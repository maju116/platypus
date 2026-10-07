# A message may not suggest something its reader cannot do.
#
# `evaluate_classes()` refused a segmentation fit by telling the reader that
# "`summarise_cases()` summarises it", and `summarise_cases` does not exist in this
# package - it is the engine's name for what R spells `summary()`. Nothing caught it: the
# refusal fires, the sentence is grammatical, and the function it names is somebody's to
# define later. The same defect was found in the engine the same afternoon, where a
# refusal named `available_transforms()` and that name was not importable.
#
# The strings are taken from the bodies of the installed functions rather than from the
# sources, for two reasons. `R CMD check` runs with no `R/` directory, so a test that
# parses the sources passes by looking at nothing there - the shape of mistake this
# project has made five times. And walking the parsed body is structural: a `#` comment
# never reaches it, which matters because `components.R` explains the `loss_` prefixes by
# saying a `dice()` that is a loss cannot coexist with a `dice()` that is a metric, and
# neither `dice()` is a function anybody is being told to call.

string_literals <- function(expression) {
  if (is.character(expression)) return(expression)
  if (!is.recursive(expression)) return(character())
  unlist(lapply(as.list(expression), string_literals), use.names = FALSE)
}

names_messages_ask_for <- function() {
  namespace <- asNamespace("platypus")
  literals <- character()
  for (name in ls(namespace, all.names = TRUE)) {
    object <- get(name, envir = namespace)
    if (!is.function(object) || is.null(body(object))) next
    literals <- c(literals, string_literals(body(object)))
  }
  hits <- regmatches(literals, gregexpr("`[a-zA-Z._][a-zA-Z0-9._]*\\(\\)`", literals))
  unique(gsub("`|\\(\\)", "", unlist(hits, use.names = FALSE)))
}

test_that("every function a message tells the reader to call exists", {
  named <- names_messages_ask_for()

  # The scan has to find something, or it passes because it looked at nothing.
  expect_gt(length(named), 10)

  defined <- c(
    ls(asNamespace("platypus"), all.names = TRUE),
    getNamespaceExports("platypus"),
    # Generics the package provides methods for, which a message may fairly name.
    "summary", "print", "plot", "predict", "as.list"
  )

  missing <- setdiff(named, defined)
  expect_identical(
    missing, character(0),
    info = paste("messages name these and nothing defines them:",
                 paste(missing, collapse = ", "))
  )
})
