#!/usr/bin/env Rscript
# Fix the links in the rendered site. Runs on `docs/` after a single Quarto render, which
# is why there is only one: polishing the generated `.qmd` instead would mean a second
# render, measured at about 160 seconds, for two things this does not need.
#
# TWO PROBLEMS, both measured on this package's 58 pages.
#
# 1. 186 LINKS ON 50 PAGES ARE ABSOLUTE URLS INTO THIS SITE.
#    Quarto's `code-link: true` resolves function names through downlit, which reads the
#    package's *published* index - so `platypus_fit()` in a code block comes out as
#    `https://maju116.github.io/platypus/<somewhere>/platypus_fit.html`, and `<somewhere>`
#    is whatever the published index currently says.
#
#    That is not a stable thing to depend on, measured twice on the same day: while pkgdown
#    was published it was `/reference/`, a path this site does not serve, so all 186 would
#    have 404'd at the switch. After the first Quarto deploy the published index said
#    `/man/` and the same 186 links pointed there instead - correct, by luck of ordering.
#
#    So the rule is not "rewrite /reference/" - that pattern was fitted to the state of the
#    world at one moment and reported 0 the next day while leaving 186 absolute links in
#    place. The rule is that **no page may link to this site by absolute URL**: relative
#    links are correct whatever is published, and they also make the site work opened from
#    disk or served anywhere else.
#
# 2. 81 MENTIONS IN PROSE ARE NOT LINKS AT ALL.
#    `[platypus_fit()]` in roxygen survives Rd as `\code{\link{}}` but altdoc's conversion
#    drops the link and keeps the code span, so the prose reads `<code>platypus_fit()</code>`.
#    §4r took these from 0 to 42 by switching roxygen's markdown on; this keeps them.
#
#    First mention per page, outside `<pre>` and outside the navigation. Thirteen links to
#    the same place on one page is noise, and the sidebar already lists every topic.

topics <- sub("[.]Rd$", "", list.files("man", pattern = "[.]Rd$"))
aliases <- list()
for (topic in topics) {
  lines <- readLines(file.path("man", paste0(topic, ".Rd")), warn = FALSE)
  for (alias in sub("^\\\\alias\\{(.*)\\}$", "\\1", grep("^\\\\alias\\{", lines, value = TRUE))) {
    aliases[[alias]] <- topic
  }
}
if (!length(aliases)) stop("no aliases found in man/ - is this the package root?")

pages <- list.files("docs", pattern = "[.]html$", recursive = TRUE, full.names = TRUE)
if (!length(pages)) stop("no docs/**.html - render the site first")

rewritten <- 0L
linked <- 0L

for (page in pages) {
  html <- paste(readLines(page, warn = FALSE), collapse = "\n")
  original <- html
  depth <- length(strsplit(sub("^docs/", "", page), "/")[[1]]) - 1L
  up <- if (depth > 0) strrep("../", depth) else ""

  # --- 1. any absolute URL into this site -> site-relative
  pattern <- 'href="https://maju116\\.github\\.io/platypus/([^"]*)"'
  hits <- gregexpr(pattern, html)
  n <- if (hits[[1]][1] == -1L) 0L else length(hits[[1]])
  if (n) {
    # `/reference/` and `/articles/` are what pkgdown served; this site puts the same
    # material under `/man/` and `/vignettes/`, so a link written against the old index is
    # redirected here rather than left to the stub files.
    html <- gsub(pattern, sprintf('href="%s\\1"', up), html)
    html <- gsub(sprintf('href="%sreference/', up), sprintf('href="%sman/', up), html, fixed = TRUE)
    html <- gsub(sprintf('href="%sarticles/', up), sprintf('href="%svignettes/', up), html, fixed = TRUE)
    rewritten <- rewritten + n
  }

  # --- 2. first unlinked mention per page becomes a link.
  # The navigation and every <pre> are cut out of a working copy first, so a name inside an
  # example is left alone and a sidebar entry is not counted as prose. The replacement then
  # happens on the real text, one mention at a time, so nothing shifts under the offsets.
  prose <- gsub("<nav.*?</nav>", "", html, perl = TRUE)
  prose <- gsub("<pre.*?</pre>", "", prose, perl = TRUE)
  found <- regmatches(prose, gregexpr("<code>[a-zA-Z._][a-zA-Z0-9._]*\\(\\)</code>", prose))[[1]]

  self <- sub("[.]html$", "", basename(page))
  seen <- character()
  for (span in found) {
    name <- gsub("<code>|\\(\\)</code>", "", span)
    topic <- aliases[[name]]
    if (is.null(topic) || identical(topic, self) || name %in% seen) next
    html <- sub(span, sprintf('<a href="%sman/%s.html">%s</a>', up, topic, span), html, fixed = TRUE)
    seen <- c(seen, name)
    linked <- linked + 1L
  }

  if (!identical(html, original)) writeLines(html, page)
}

cat(sprintf("polished %d pages: %d absolute links made relative, %d prose mentions linked\n",
            length(pages), rewritten, linked))

# The check that matters, run here rather than left to a reader: no page may link to this
# site by absolute URL, whatever the published index happens to say today.
left <- 0L
for (page in pages) {
  html <- paste(readLines(page, warn = FALSE), collapse = "\n")
  found <- gregexpr('href="https://maju116\\.github\\.io/platypus/', html)[[1]]
  if (found[1] != -1L) left <- left + length(found)
}
if (left) stop(sprintf("%d links into this site are still absolute", left))
cat("no page links into this site by absolute URL\n")
