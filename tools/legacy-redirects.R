#!/usr/bin/env Rscript
# Keep the links people already have.
#
# This package's site was pkgdown until 2026-10-07, which published 99 pages under
# `/reference/` and three articles under `/articles/`. The Quarto site serves the same
# material from `/man/` and `/vignettes/`, so without these every one of those addresses
# 404s - and they are not only in our own README history: the repository has 134 stars and
# 28 open issues, and other people's comments carry those links.
#
# The alternative to a redirect is leaving the old pkgdown copy published beside the new
# one, which is what `clean: false` was doing. That is worse: two complete copies of the
# documentation, one of them never updated again, both indexed. A page that quietly drifts
# from the truth is worse than a page that is gone, and a redirect is better than either.
#
# The same judgement as §4o, which kept `master` and `@0.1.1` working for the forks and the
# issues rather than force-pushing over them.
#
# Mapping, in order:
#   1. an alias from man/*.Rd -> the topic that documents it now. This is most of the work:
#      pkgdown gave `loss_dice`, `callback_early_stopping` and `linknet` pages of their
#      own, while here they are aliases of `losses`, `callbacks` and `models`.
#   2. the eight functions renamed on 2026-10-07 -> their new topic.
#   3. the three articles -> the same vignette under its new directory.
#   4. the index pages -> the site root.
#
# Delete this script when the old addresses stop appearing in the logs. Until then it is
# 50 lines and a few hundred bytes per file.

topics <- sub("[.]Rd$", "", list.files("man", pattern = "[.]Rd$"))
alias_to_topic <- list()
for (topic in topics) {
  lines <- readLines(file.path("man", paste0(topic, ".Rd")), warn = FALSE)
  for (alias in sub("^\\\\alias\\{(.*)\\}$", "\\1", grep("^\\\\alias\\{", lines, value = TRUE))) {
    alias_to_topic[[alias]] <- topic
  }
}

# Renamed on 2026-10-07, so an old link names something that no longer exists anywhere.
renamed <- c(
  platypus_split = "split_dataset",
  split_files = "split_path",
  mask_report = "inspect_masks",
  mask_classes = "colours_to_classes",
  mask_colours = "classes_to_colours",
  save_weights = "export_weights",
  available_augmentations = "available_transforms",
  augment = "augmentation_step"
)

stub <- function(target, what) {
  c(
    "<!DOCTYPE html>",
    "<html lang=\"en\"><head><meta charset=\"utf-8\">",
    sprintf("<meta http-equiv=\"refresh\" content=\"0; url=%s\">", target),
    sprintf("<link rel=\"canonical\" href=\"%s\">", target),
    "<meta name=\"robots\" content=\"noindex\">",
    sprintf("<title>Moved: %s</title>", what),
    "</head><body>",
    sprintf("<p>This page moved. <a href=\"%s\">%s</a></p>", target, what),
    "<p>The site was rebuilt with Quarto on 2026-10-07; the reference now lives under",
    "<code>/man/</code> and the articles under <code>/vignettes/</code>.</p>",
    "</body></html>"
  )
}

written <- 0L
dir.create("docs/reference", recursive = TRUE, showWarnings = FALSE)

# every alias pkgdown would have given a page, plus the renames
old_names <- unique(c(names(alias_to_topic), names(renamed)))
for (old in old_names) {
  topic <- if (old %in% names(renamed)) renamed[[old]] else alias_to_topic[[old]]
  if (is.null(topic) || !file.exists(file.path("docs", "man", paste0(topic, ".html")))) next
  writeLines(stub(sprintf("../man/%s.html", topic), sprintf("%s()", old)),
             file.path("docs", "reference", paste0(old, ".html")))
  written <- written + 1L
}

# the reference index, which had no single successor - the sidebar is the index now
writeLines(stub("../index.html", "the reference"), "docs/reference/index.html")
written <- written + 1L

# articles -> vignettes
dir.create("docs/articles", recursive = TRUE, showWarnings = FALSE)
for (vignette in sub("[.]html$", "", list.files("docs/vignettes", pattern = "[.]html$"))) {
  writeLines(stub(sprintf("../vignettes/%s.html", vignette), vignette),
             file.path("docs", "articles", paste0(vignette, ".html")))
  written <- written + 1L
}
writeLines(stub("../index.html", "the articles"), "docs/articles/index.html")
written <- written + 1L

# pkgdown's own top-level pages that have no counterpart here
for (page in c("authors.html", "LICENSE-text.html")) {
  writeLines(stub("index.html", "the site"), file.path("docs", page))
  written <- written + 1L
}

# pkgdown wrote this and altdoc does not; without it `clean: true` removes it and GitHub
# Pages starts running Jekyll over the output.
invisible(file.create("docs/.nojekyll"))

cat(sprintf("wrote %d redirects, and docs/.nojekyll\n", written))
