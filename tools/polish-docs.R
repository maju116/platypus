#!/usr/bin/env Rscript
# Fix the links in the rendered site. Runs on `docs/` after a single Quarto render, which
# is why there is only one: polishing the generated `.qmd` instead would mean a second
# render, measured at about 160 seconds, for two things this does not need.
#
# TWO PROBLEMS, both measured on this package's 58 pages.
#
# 1. 186 LINKS ON 50 PAGES POINT AT THE OLD PKGDOWN SITE.
#    Quarto's `code-link: true` resolves function names through downlit, which reads the
#    package's published pkgdown index - so `platypus_fit()` in a code block became
#    `https://maju116.github.io/platypus/reference/platypus_fit.html`. That path is served
#    today and will not be once this site replaces it: the reference lives at `/man/` here.
#    Every one of those 186 would have become a 404 on the day of the switch, with a green
#    build and nothing to say so.
#
#    They are rewritten to site-relative paths, which also means the site works when opened
#    from disk or served anywhere - the absolute URL was a second thing to get wrong.
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

  # --- 1. absolute pkgdown URLs -> site-relative
  hits <- gregexpr('href="https://maju116\\.github\\.io/platypus/reference/[a-zA-Z0-9._]+\\.html"', html)
  n <- if (hits[[1]][1] == -1L) 0L else length(hits[[1]])
  if (n) {
    html <- gsub(
      'href="https://maju116\\.github\\.io/platypus/reference/([a-zA-Z0-9._]+)\\.html"',
      sprintf('href="%sman/\\1.html"', up),
      html
    )
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

cat(sprintf("polished %d pages: %d absolute pkgdown links rewritten, %d prose mentions linked\n",
            length(pages), rewritten, linked))

# The check that matters, run here rather than left to a reader: nothing may still point at
# the site this one replaces.
left <- 0L
for (page in pages) {
  html <- paste(readLines(page, warn = FALSE), collapse = "\n")
  left <- left + length(grep("maju116\\.github\\.io/platypus/reference/", html))
}
if (left) stop(sprintf("%d links still point at the old pkgdown paths", left))
cat("no link points at the old pkgdown reference paths\n")
