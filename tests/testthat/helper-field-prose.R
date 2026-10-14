#' The engine's own prose for every configuration field, digested
#'
#' One entry per field *name*, not per field-and-definition. 92 names carry a description
#' across 42 definitions and 201 pairs, and 17 of those names are described differently
#' depending on where they appear - `learning_rate` under Adam and under SGD, `alpha` in
#' seven losses. A reader meets the name once, through one `@param`, so the digest covers
#' every description that name has: a rewrite of any of them is a reason to re-read the one
#' R text.
#'
#' Keyed by the digest rather than by the text so the fixture stays a ledger. The text
#' itself is 24 kB and would turn a question - does R's prose still hold? - into a copy of
#' the engine that nothing could tell apart from the original.
#'
#' @keywords internal
#' @noRd
described_field_prose <- function(schema = NULL) {
  schema <- schema %||% rplatypus:::engine()$spec_schema()
  found <- list()
  for (definition in schema$`$defs`) {
    properties <- definition$properties
    if (!length(properties)) next
    for (name in names(properties)) {
      description <- trimws(properties[[name]]$description %||% "")
      if (nzchar(description)) found[[name]] <- c(found[[name]], description)
    }
  }
  vapply(found[order(names(found))], function(texts) {
    # `method = "radix"` because the default sorts by the locale's collation, and the digest
    # must not. Measured here: `architecture` carries two descriptions, one starting "Which
    # detector" and one "Which U-shaped network", and en_US.UTF-8 orders them the other way
    # round from C - so the generator under Rscript and the test under testthat, which sets
    # LC_COLLATE=C, disagreed about the same two strings on the same machine. A digest that
    # depends on where it is computed is §3C's unportable number in another costume.
    rlang::hash(sort(unique(texts), method = "radix"))
  }, character(1))
}

#' The recorded ledger, from wherever the tests are being run
#'
#' @keywords internal
#' @noRd
recorded_field_prose <- function() {
  for (root in c(".", "..", file.path("..", ".."))) {
    path <- file.path(root, "tests", "testthat", "fixtures", "field-prose.yml")
    if (!file.exists(path)) path <- file.path(root, "fixtures", "field-prose.yml")
    if (file.exists(path)) return(yaml::read_yaml(path))
  }
  NULL
}
