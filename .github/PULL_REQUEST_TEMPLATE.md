<!--
Shared in shape with pyplatypus/.github/PULL_REQUEST_TEMPLATE.md: the same three headings,
each side listing its own checks. Nothing here is a gate you must pass before opening a pull
request - a draft with half of it done is welcome, and CI runs the same checks anyway.

The one that is not obvious is the last. A test that cannot fail reads as proof and is none,
so the ask is to break the thing it guards and watch it go red. It takes a minute and it has
caught more than it has cost.
-->

## What this changes

## Closes

<!-- `Closes #123`, or say which issue it is part of. -->

## How it was checked

- [ ] `Rscript -e 'pkgload::load_all(); testthat::test_dir("tests/testthat")'` - and **0 skipped**, not just 0 failed
- [ ] `Rscript -e 'lintr::lint_package()'` - 0 findings
- [ ] `R CMD build . && R CMD check rplatypus_*.tar.gz --no-manual` - the way it ships, vignettes included
- [ ] the installed package, not the sources: `library(rplatypus)` + `testthat::test_dir()`, never `load_all()`
- [ ] every new test was mutated to prove it can fail
- [ ] documentation updated, if an exported name, an argument or a number in prose changed

<!--
A skipped test is the signal, not the noise: against a working engine a new test that skips
means the pin or the probe is wrong, and it reads exactly like a machine with no Python.

The engine's version is pinned in R/bridge.R. If this pull request needs an unreleased
pyplatypus, say so - it cannot merge before that version is on PyPI.
-->
