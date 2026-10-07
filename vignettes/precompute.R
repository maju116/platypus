# The vignettes that run something are precomputed.
#
# `configuration.Rmd` is not among them and has no `.Rmd.orig`: it is prose, YAML and R chunks
# with `eval = FALSE`, so there is nothing to run and nothing to keep in step. Its claims are
# checked by tests/testthat/test-guide.R instead, which validates every configuration it shows
# through the engine and asserts every function it names exists.
#
# R CMD check builds vignettes, and one of these trains two models on 5.6 GB of images that a
# CRAN machine does not have and would not want. So the real documents are the .Rmd.orig files,
# knitted here into the .Rmd that ship - which means every number and every figure in a vignette
# came from a real run, rather than being described in prose and hoped for.
#
# Run from the package root:
#
#     PLATYPUS_DSBOWL=/path/to/data_science_bowl \
#     PLATYPUS_BCCD=/path/to/BCCD Rscript vignettes/precompute.R
#
# The volume vignette generates its own data, so it needs nothing but a working engine. Name
# one or more vignettes as arguments to knit only those.
#
# The blood-cell one trains a 61.5-million-parameter detector for 150 epochs - about 37
# minutes on a GTX 1070 - so it is the slow one.

which <- commandArgs(trailingOnly = TRUE)
if (!length(which)) which <- c("data-science-bowl", "volumes", "blood-cells")

if ("data-science-bowl" %in% which) {
  stopifnot(nzchar(Sys.getenv("PLATYPUS_DSBOWL")))
}
if ("blood-cells" %in% which) {
  stopifnot(nzchar(Sys.getenv("PLATYPUS_BCCD")))
}

# Against the sources, not whatever is installed. The first run of the volume vignette knitted
# happily against platypus 0.2.0 from the user library: half the functions did not exist yet, and
# knitr pasted "could not find function" into the document instead of failing.
pkgload::load_all(".", quiet = TRUE)

withr::with_dir("vignettes", {
  for (name in which) {
    message("knitting ", name)
    knitr::knit(paste0(name, ".Rmd.orig"), paste0(name, ".Rmd"))
  }
})
