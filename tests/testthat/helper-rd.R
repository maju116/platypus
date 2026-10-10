#' The package's own Rd database, from the source tree when it is reachable
#'
#' Source `man/` first, installed package second, and the order matters.
#'
#' `tools::Rd_db("platypus")` reads the *installed* package - correct under `R CMD check`,
#' which has just built one, and empty under `pkgload::load_all()`, which is how this package
#' is most often worked on locally. Reading only the installed database left a test skipping
#' in exactly that case: 2 skipped where 0 was expected, and a silently inert tripwire is
#' worse than none. The same hazard made a vignette knit against platypus 0.2.0 and a linter
#' report live helpers as undefined - and caught the author again on 2026-10-10, when a test
#' running through `callr` read an installed package still carrying the previous pin.
#'
#' @keywords internal
#' @noRd
rd_database <- function() {
  for (root in c(".", "..", file.path("..", ".."))) {
    if (dir.exists(file.path(root, "man")) && file.exists(file.path(root, "DESCRIPTION"))) {
      db <- tryCatch(tools::Rd_db(dir = root), error = function(e) list())
      if (length(db)) return(db)
    }
  }
  tryCatch(tools::Rd_db("platypus"), error = function(e) list())
}

#' Every name documented as an argument, anywhere in the package
#'
#' Read from the Rd rather than from `#' @param` in `R/`, because the Rd is what a reader
#' sees. A roxygen comment that never regenerated would pass a grep over the source and
#' still be absent from `?platypus_spec`.
#'
#' @keywords internal
#' @noRd
documented_arguments <- function(db = rd_database()) {
  tag_of <- function(x) {
    tag <- attr(x, "Rd_tag")
    if (is.null(tag)) "" else tag
  }
  found <- character(0)
  for (rd in db) {
    tags <- vapply(rd, tag_of, character(1))
    for (block in rd[tags == "\\arguments"]) {
      items <- block[vapply(block, tag_of, character(1)) == "\\item"]
      for (item in items) {
        # An \item is a pair: the names, then the description. Several arguments can share
        # one entry - `validation, split, test` - so the first part is split on commas.
        text <- paste(unlist(lapply(item[[1]], as.character)), collapse = "")
        found <- c(found, trimws(strsplit(text, ",")[[1]]))
      }
    }
  }
  unique(found[nzchar(found)])
}
