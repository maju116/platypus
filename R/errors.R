# Turning the engine's complaints into R conditions.
#
# The audience is someone who knows their data and their question, and does not know or
# care that any of this is Python. A traceback tells them nothing; "models[1]$loss: no
# such loss 'focaal', did you mean 'focal'?" tells them everything.
#
# pyplatypus reports its problems as plain data for exactly this reason, and the shim
# hands that data over intact rather than letting reticulate flatten it into a string.

#' Turn the engine's problem report into an R condition and raise it
#'
#' @param failure The list returned by the shim when `ok` is `FALSE`.
#' @param call The call to blame, for the traceback.
#' @keywords internal
#' @noRd
abort_engine <- function(failure, call = sys.call(-1)) {
  problems <- failure$problems %||% list()

  lines <- vapply(problems, function(p) {
    where <- p$where %||% "(top level)"
    text <- paste0("  * ", r_path(where), ": ", p$problem %||% "invalid")
    # `got` earns its place when it is the offending value - `2.0`, `'focaal'`. When the
    # problem is with a whole model, the offending value is the whole model, and echoing
    # a truncated dump of it back at the reader helps nobody.
    if (!is.null(p$got) && nchar(p$got) <= 60) {
      text <- paste0(text, " (got ", p$got, ")")
    }
    text
  }, character(1))

  headline <- if (length(problems) == 1L) {
    "The specification has a problem:"
  } else if (length(problems) > 1L) {
    paste0("The specification has ", length(problems), " problems:")
  } else {
    failure$message %||% "The engine rejected the specification."
  }

  message <- if (length(lines)) paste(c(headline, lines), collapse = "\n") else headline

  stop(structure(
    class = c(paste0("platypus_", failure$kind %||% "error"), "platypus_error",
              "error", "condition"),
    list(message = message, call = call,
         problems = problems_frame(problems), source = failure$source)
  ))
}

#' Rewrite a Python path so it reads like the R that produced it
#'
#' `models[0].loss` is where pydantic found the problem, but the user wrote
#' `models = list(u_net(...))`, so `models[1]$loss` is where they should look. Zero-based
#' indices in an R error message are a small cruelty.
#' @keywords internal
#' @noRd
r_path <- function(where) {
  if (!is.character(where) || !nzchar(where)) return("(top level)")
  bumped <- gsub("\\[([0-9]+)\\]", "\u0001\\1\u0002", where)
  repeat {
    m <- regmatches(bumped, regexpr("\u0001[0-9]+\u0002", bumped))
    if (!length(m) || !nzchar(m)) break
    n <- as.integer(gsub("[\u0001\u0002]", "", m))
    bumped <- sub("\u0001[0-9]+\u0002", paste0("[[", n + 1L, "]]"), bumped)
  }
  gsub("\\.", "$", bumped)
}

#' The problems as a data frame, for anyone who wants to work with them
#' @keywords internal
#' @noRd
problems_frame <- function(problems) {
  if (!length(problems)) {
    return(data.frame(where = character(), problem = character(),
                      got = character(), stringsAsFactors = FALSE))
  }
  data.frame(
    where = vapply(problems, function(p) r_path(p$where %||% ""), character(1)),
    problem = vapply(problems, function(p) p$problem %||% NA_character_, character(1)),
    got = vapply(problems, function(p) p$got %||% NA_character_, character(1)),
    stringsAsFactors = FALSE
  )
}

`%||%` <- function(x, y) if (is.null(x)) y else x
