# Vignettes here are precomputed.
#
# R CMD check builds vignettes, and one of these trains two models on 5.6 GB of images that a
# CRAN machine does not have and would not want. So the real documents are the .Rmd.orig files,
# knitted here into the .Rmd that ship - which means every number and every figure in a vignette
# came from a real run, rather than being described in prose and hoped for.
#
# Run from the package root:
#
#     PLATYPUS_DSBOWL=/path/to/data_science_bowl Rscript vignettes/precompute.R
#
# The volume vignette generates its own data, so it needs nothing but a working engine. Pass
# `volumes` as an argument to knit only that one.

which <- commandArgs(trailingOnly = TRUE)
if (!length(which)) which <- c("data-science-bowl", "volumes")

if ("data-science-bowl" %in% which) {
  stopifnot(nzchar(Sys.getenv("PLATYPUS_DSBOWL")))
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
