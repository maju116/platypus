# The issue forms and the pull-request template, in the shape both repositories share.
#
# Same invariant as the README and the reference groups, and here for the same reason: the two
# cannot be equal - only a user of this package can be unsure which half they are in, and only
# the engine can ask for a torch version - but they can ask the same things in the same order,
# so a contributor who has filed in one repository is not starting again in the other.
#
# Why this is a script in `tools/` rather than a test in `tests/testthat/`: `.github` is in
# `.Rbuildignore`, so a test reading it would skip under `R CMD check` - and §3D.7 of the
# instructions makes the skip count the signal, which means a permanent, expected skip costs
# more than this file does. The engine has no tarball, so over there it is an ordinary test.
#
# Run from the package root:
#
#     Rscript tools/check-github-templates.R

templates <- ".github/ISSUE_TEMPLATE"
pull_request <- ".github/PULL_REQUEST_TEMPLATE.md"

# Also in pyplatypus's tests/test_github_templates.py. `half` is this side's alone - a user
# of the engine is already in the engine - so each side checks a subsequence, never equality.
bug_field_order <- c(
  "what-happened",
  "half",
  "expected",
  "reproduce",
  "session",
  "anything-else"
)

feature_field_order <- c(
  "problem",
  "proposal",
  "alternatives",
  "measurement",
  "anything-else"
)

# The subsequence check permits absence, which is how drift gets back in, so the fields that
# must be present are named separately. `session` is the whole point of the exercise.
bug_required <- c("what-happened", "half", "expected", "reproduce", "session")
feature_required <- c("problem", "proposal")

pull_request_headings <- c("What this changes", "Closes", "How it was checked")

# An environment rather than `<<-`, which lintr refuses and which this package has already
# replaced once, in `predict.platypus_fit()`: the superassignment there was crossing the
# Python boundary and the bug it caused was worth the lesson.
found <- new.env(parent = emptyenv())
found$failures <- character(0)
note <- function(...) {
  found$failures <- c(found$failures, paste0(...))
}

field_ids <- function(form) {
  parsed <- yaml::read_yaml(file.path(templates, form))
  ids <- vapply(parsed$body, function(block) {
    if (is.null(block$id)) NA_character_ else block$id
  }, character(1))
  ids[!is.na(ids)]
}

required_ids <- function(form) {
  parsed <- yaml::read_yaml(file.path(templates, form))
  keep <- vapply(parsed$body, function(block) {
    isTRUE(block$validations$required) && !is.null(block$id)
  }, logical(1))
  vapply(parsed$body[keep], function(block) block$id, character(1))
}

# What a reporter is shown, rather than what the file says to whoever maintains it: the
# comments in these forms record the mistake being fixed, and naming it is not asking it.
shown_text <- function(path) {
  if (grepl("\\.md$", path)) {
    text <- paste(readLines(path, warn = FALSE), collapse = "\n")
    return(gsub("<!--.*?-->", "", text))
  }
  paste(utils::capture.output(print(yaml::read_yaml(path))), collapse = "\n")
}

for (pair in list(
  list(form = "bug_report.yml", order = bug_field_order, must = bug_required),
  list(form = "feature_request.yml", order = feature_field_order, must = feature_required)
)) {
  got <- field_ids(pair$form)
  if (!length(got)) note(pair$form, " has no identified fields")

  extra <- setdiff(got, pair$order)
  if (length(extra)) {
    note(
      pair$form, " has fields outside the shared order: ", paste(extra, collapse = ", "),
      ". Adding one means adding it here and in pyplatypus's tests/test_github_templates.py"
    )
  }

  expected <- pair$order[pair$order %in% got]
  if (!identical(got, expected)) {
    note(
      pair$form, " is out of the order both repositories share: ",
      paste(got, collapse = ", ")
    )
  }

  missing <- setdiff(pair$must, got)
  if (length(missing)) {
    note(pair$form, " stopped asking: ", paste(missing, collapse = ", "))
  }
}

# A markdown section gets deleted; `required: true` does not, which is why these are forms.
if (!"session" %in% required_ids("bug_report.yml")) {
  note("the session field is no longer required, which is the field that was always missing")
}

prefilled <- {
  parsed <- yaml::read_yaml(file.path(templates, "bug_report.yml"))
  block <- Filter(function(b) identical(b$id, "session"), parsed$body)[[1]]
  block$attributes$value
}
for (wanted in c("sessionInfo()", "platypus_status()")) {
  if (!grepl(wanted, prefilled, fixed = TRUE)) {
    note("the session field no longer prefills ", wanted)
  }
}

# The regression. Both repositories asked for a TensorFlow version until 2026-10-11, and this
# one told the reporter to run `tensorflow::tf_version()`, which this package does not have.
#
# What is searched is the **questions** - the body of each form - and not `config.yml`, whose
# contact links deliberately name the 2020 TensorFlow package: that is where the stars and the
# open issues still are, and sending somebody there is the opposite of the mistake. The first
# version of this check flagged that link, which is a rule calibrated against the word rather
# than against the error. `tf_version` is still matched anywhere, since nothing but the old
# template has a reason to say it.
for (form in list.files(templates, pattern = "_(report|request)\\.yml$")) {
  parsed <- yaml::read_yaml(file.path(templates, form))
  asked <- tolower(paste(utils::capture.output(print(parsed$body)), collapse = "\n"))
  if (grepl("tensorflow", asked)) note(form, " still asks about TensorFlow")
}
for (path in c(list.files(templates, pattern = "\\.yml$", full.names = TRUE), pull_request)) {
  if (grepl("tf_version", shown_text(path), fixed = TRUE)) {
    note(basename(path), " still tells the reporter to run tensorflow::tf_version()")
  }
}

# §3G routes issues by where the change is visible, so the wrong repository is the predictable
# failure, and a contact link is the only place GitHub shows that choice.
links <- yaml::read_yaml(file.path(templates, "config.yml"))$contact_links
if (!any(grepl("maju116/pyplatypus", vapply(links, function(l) l$url %||% "", character(1))))) {
  note("no contact link points at the engine")
}

pr_text <- paste(readLines(pull_request, warn = FALSE), collapse = "\n")
headings <- trimws(sub("^## ", "", grep("^## ", strsplit(pr_text, "\n")[[1]], value = TRUE)))
if (!identical(headings, pull_request_headings)) {
  note(
    "the shared pull-request headings changed on this side only: ",
    paste(headings, collapse = ", ")
  )
}
if (!grepl("mutated to prove it can fail", pr_text, fixed = TRUE)) {
  note("the pull-request template dropped the one check that is this project's own")
}

if (length(found$failures)) {
  cat("github templates: ", length(found$failures), " problem(s)\n", sep = "")
  cat(paste0("  - ", found$failures, collapse = "\n"), "\n", sep = "")
  quit(status = 1)
}
cat("github templates: the shared order holds, on both forms and the pull request\n")
