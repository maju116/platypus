# Vignettes here are precomputed.
#
# R CMD check builds vignettes, and this one trains two models on 5.6 GB of images that
# a CRAN machine does not have and would not want. So the real document is the .Rmd.orig,
# knitted here into the .Rmd that ships - which means every number and every figure in the
# vignette came from a real run, rather than being described in prose and hoped for.
#
# Run from the package root, with the data to hand:
#
#     PLATYPUS_DSBOWL=/path/to/data_science_bowl Rscript vignettes/precompute.R

stopifnot(nzchar(Sys.getenv("PLATYPUS_DSBOWL")))
withr::with_dir("vignettes", {
  knitr::knit("data-science-bowl.Rmd.orig", "data-science-bowl.Rmd")
})
