# The bridge to the engine.
#
# Everything this package does eventually happens in Python. The point of this file is
# that a user never has to know that: no interpreter to choose, no environment to
# activate, no version to match. reticulate::py_require() declares what is needed and uv
# builds an isolated environment for it on first use.
#
# That isolation is the whole design. The R keras and tensorflow packages of 2020 shared
# one Python installation with whatever else the user had, so an unrelated change on the
# Python side could break an R session. Nothing here touches a Python anyone else uses.

#' The exact engine this version of platypus speaks to
#'
#' Pinned rather than ranged on purpose: pyplatypus is an alpha whose API still moves,
#' and an R package should not break because a Python dependency drifted underneath it.
#' @keywords internal
#' @noRd
PYPLATYPUS_REQUIREMENT <- "pyplatypus==0.2.0a1"

# Holds the imported module. Populated lazily, so loading this package costs nothing.
.platypus <- new.env(parent = emptyenv())

#' Complain about RETICULATE_PYTHON before it complains about us
#'
#' It overrides py_require() completely, so a stale value turns every function in this
#' package into an unreadable failure somewhere inside reticulate. This was not
#' hypothetical: the author's own machine carried a RETICULATE_PYTHON pointing at an
#' Anaconda deleted years earlier, and it broke the first run of the prototype.
#' @keywords internal
#' @noRd
check_reticulate_python <- function() {
  value <- Sys.getenv("RETICULATE_PYTHON", unset = NA_character_)
  if (is.na(value) || !nzchar(value)) {
    return(invisible(NULL))
  }
  if (!file.exists(value)) {
    packageStartupMessage(
      "platypus: RETICULATE_PYTHON points at '", value, "', which does not exist.\n",
      "  That setting overrides the isolated environment this package manages, so ",
      "nothing here will work\n  until it is removed. Look in ~/.Renviron and ~/.bashrc, ",
      "then restart R."
    )
  } else {
    packageStartupMessage(
      "platypus: RETICULATE_PYTHON is set to '", value, "'.\n",
      "  That Python will be used instead of the isolated environment this package ",
      "manages, so it must\n  have pyplatypus installed. Unset it to let platypus ",
      "handle this itself."
    )
  }
  invisible(NULL)
}

.onLoad <- function(libname, pkgname) {
  check_reticulate_python()
  reticulate::py_require(PYPLATYPUS_REQUIREMENT)

  .platypus$engine <- reticulate::import(
    "pyplatypus",
    delay_load = list(
      # Nothing is downloaded until a function actually needs Python, so library(platypus)
      # stays instant and works offline.
      on_error = function(e) {
        stop(
          "platypus could not start its Python engine.\n",
          "  ", conditionMessage(e), "\n\n",
          "  The first call needs to download ", PYPLATYPUS_REQUIREMENT, " and PyTorch, ",
          "which needs a network\n  connection once. After that it is cached and works ",
          "offline. See `?platypus_status`.",
          call. = FALSE
        )
      }
    )
  )
  invisible(NULL)
}

#' The Python engine, imported on first use
#' @keywords internal
#' @noRd
engine <- function() {
  .platypus$engine
}

#' The R-facing shim
#'
#' A small Python module shipped with this package. It exists because exceptions do not
#' survive the crossing usefully - reticulate hands R the message and drops the object -
#' so everything it does returns data instead of raising. See inst/python.
#' @keywords internal
#' @noRd
shim <- function() {
  if (is.null(.platypus$shim)) {
    .platypus$shim <- reticulate::import_from_path(
      "platypus_shim",
      path = system.file("python", package = "platypus", mustWork = TRUE)
    )
  }
  .platypus$shim
}

#' Is the engine ready?
#'
#' Reports whether the Python side has been started, and what it is. Calling this does
#' not start it - use it to check before a long run, or when something is wrong.
#'
#' @return A list with the engine version, the interpreter in use and where its
#'   environment lives, or `NULL` fields when Python has not been started yet.
#' @export
#' @examples
#' \dontrun{
#' platypus_status()
#' }
platypus_status <- function() {
  started <- reticulate::py_available(initialize = FALSE)
  out <- list(
    requirement = PYPLATYPUS_REQUIREMENT,
    started = started,
    reticulate = as.character(utils::packageVersion("reticulate")),
    reticulate_python = Sys.getenv("RETICULATE_PYTHON", unset = NA_character_)
  )
  if (started) {
    config <- reticulate::py_config()
    out$python <- config$python
    out$python_version <- as.character(config$version)
    out$environment <- config$virtualenv
    out$engine_version <- tryCatch(engine()$`__version__`, error = function(e) NA_character_)
  }
  structure(out, class = "platypus_status")
}

#' @export
print.platypus_status <- function(x, ...) {
  cat("platypus engine\n")
  cat("  requires    :", x$requirement, "\n")
  cat("  started     :", if (isTRUE(x$started)) "yes" else "no (starts on first use)", "\n")
  if (isTRUE(x$started)) {
    cat("  version     :", x$engine_version, "\n")
    cat("  python      :", x$python, paste0("(", x$python_version, ")"), "\n")
    cat("  environment :", x$environment, "\n")
  }
  if (!is.na(x$reticulate_python)) {
    cat("  note        : RETICULATE_PYTHON is set, overriding the managed environment\n")
  }
  invisible(x)
}
