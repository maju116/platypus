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
PYPLATYPUS_VERSION <- "0.3.0a8"

#' What this session will ask for. Set by [platypus_use_torch()] before the engine starts.
#' @keywords internal
#' @noRd
.platypus_requirement <- local({
  extra <- NULL
  function(set) {
    if (!missing(set)) extra <<- set

    # Development escape hatch. The pin is deliberate - it stops an R release breaking
    # because the engine drifted underneath it - but it also means work on the two sides
    # cannot meet until the Python half is published. Pointing this at a source tree lets
    # them meet during development, and is never used by anyone installing the package.
    local_engine <- Sys.getenv("PLATYPUS_ENGINE_PATH", unset = "")
    if (nzchar(local_engine)) {
      return(normalizePath(local_engine, mustWork = TRUE))
    }

    # `hub` is always asked for. It is optional in the Python package because most runs never
    # fetch published weights and an air-gapped one cannot - but from R the environment is built
    # by this package, torch already dominates the download by three orders of magnitude, and
    # without it `weights = "dsbowl-unet"` cannot work at all. That was not a hypothetical: the
    # vignette caught it, with the feature unusable from R in every release that had it.
    #
    # Installing it needs a network once; using it needs one only when a name is fetched.
    extras <- if (is.null(extra)) "hub" else paste(extra, "hub", sep = ",")
    sprintf("pyplatypus[%s]==%s", extras, PYPLATYPUS_VERSION)
  }
})

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

#' Explain a failure to start the engine
#'
#' Separated from the handler so it can be tested without breaking a Python installation.
#'
#' One case deserves its own paragraph. uv caches what an index told it, so for a while
#' after a new pyplatypus is published uv still believes the version this package pins
#' does not exist - and says so in a way that sounds permanent: "no version of
#' pyplatypus==x". Anyone who updates platypus the day a release goes out can meet it, and
#' nothing in the message hints that the fix is local and takes a second.
#' @keywords internal
#' @noRd
engine_start_failure <- function(message) {
  stale_index <- grepl("no version of pyplatypus", message, fixed = TRUE) ||
    grepl("No solution found when resolving", message, fixed = TRUE)

  out <- paste0(
    "platypus could not start its Python engine.\n",
    "  ", message, "\n\n"
  )
  if (stale_index) {
    paste0(
      out,
      "  ", .platypus_requirement(), " does exist. uv is answering from a cached copy of ",
      "the package\n  index, which happens for a while after a release.\n\n",
      "  Clear the index cache - not the whole cache, which would throw away the multi-",
      "gigabyte\n  PyTorch download along with it:\n\n",
      "    rm -rf \"$(uv cache dir)\"/simple-v*\n\n",
      "  `uv cache clean pyplatypus` is not enough: it removes that package's built ",
      "artifacts\n  and leaves the index response that says the version does not exist.\n"
    )
  } else {
    paste0(
      out,
      "  The first call needs to download ", .platypus_requirement(), " and PyTorch, ",
      "which needs a network\n  connection once. After that it is cached and works ",
      "offline. See `?platypus_status`.\n"
    )
  }
}

.onLoad <- function(libname, pkgname) {
  check_reticulate_python()
  reticulate::py_require(.platypus_requirement())

  .platypus$engine <- reticulate::import(
    "pyplatypus",
    delay_load = list(
      # Nothing is downloaded until a function actually needs Python, so library(platypus)
      # stays instant and works offline.
      on_error = function(e) {
        stop(engine_start_failure(conditionMessage(e)), call. = FALSE)
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
    requirement = .platypus_requirement(),
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


#' Ask for a particular PyTorch build
#'
#' Call this before anything starts the engine - straight after `library(platypus)` - and
#' before the first call that needs Python.
#'
#' There is one reason to: **a GeForce GTX 10-series card, or anything older**. PyTorch
#' 2.8 and later ship CUDA 13 builds, and CUDA 13 dropped the Maxwell, Pascal and Volta
#' generations outright. No driver update brings them back. Without this, such a card is
#' simply not used and everything runs on the processor, perhaps ten times slower, with
#' nothing obviously wrong.
#'
#' On anything from Turing - the RTX 20-series onwards - the default is correct and this
#' is unnecessary.
#'
#' @param build `"pascal"` for the last PyTorch built against CUDA 12, or `NULL` for the
#'   default.
#' @return The requirement string now in force, invisibly.
#' @export
#' @examples
#' \dontrun{
#' library(platypus)
#' platypus_use_torch("pascal")   # GTX 10-series and older
#' }
platypus_use_torch <- function(build = c("pascal", "default")) {
  build <- if (is.null(build)) "default" else match.arg(build)
  if (reticulate::py_available(initialize = FALSE)) {
    stop(
      "The engine has already started, so the PyTorch build can no longer be changed.\n",
      "  Call platypus_use_torch() right after library(platypus), before anything that ",
      "needs Python.\n  Restart R to change it now.",
      call. = FALSE
    )
  }
  .platypus_requirement(if (identical(build, "default")) NULL else build)
  reticulate::py_require(.platypus_requirement())
  invisible(.platypus_requirement())
}

#' Where the work will happen
#'
#' Reports the PyTorch build in use and the device it can see. Starts the engine if it is
#' not running.
#'
#' @return A list, printed readably.
#' @export
#' @examples
#' \dontrun{
#' platypus_device()
#' }
platypus_device <- function() {
  structure(shim()$device_report(), class = "platypus_device")
}

#' @export
print.platypus_device <- function(x, ...) {
  cat("torch      :", x$torch, "\n")
  cat("cuda build :", x$cuda_build, "\n")
  cat("device     :", x$device, "\n")
  if (isTRUE(x$gpu_present_but_unusable)) {
    cat("\n  This machine has a GPU that this PyTorch build cannot use, so the work will\n")
    cat("  run on the processor instead - often about ten times slower.\n")
    cat("  For a GTX 10-series card or older, restart R and call:\n")
    cat("      platypus_use_torch(\"pascal\")\n")
  }
  invisible(x)
}
