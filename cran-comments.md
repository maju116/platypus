# cran-comments

First submission of `rplatypus`.

## Test environments

- local: Ubuntu 24.04, R 4.5.3
- GitHub Actions, on every push: Ubuntu (R devel, release, oldrel-1), macOS (release),
  Windows (release)

## R CMD check results

`R CMD check --as-cran`, with the manual, gives one NOTE on the maintainer's machine:

    * checking HTML version of manual ... NOTE
      Skipping checking HTML validation: no command 'tidy' found.

That is a missing local tool rather than a finding. There are no ERRORs and no WARNINGs.

The expected `New submission` NOTE will also appear.

## The package name

This file previously described a conflict and promised to resubmit under another name if it
could not be resolved. **That is what this name is.** The package was called `platypus` on
GitHub from 2020 until 2026-10-14; `R CMD check --as-cran` reported a conflict with
`Platypus`, archived from CRAN on 2026-02-02 - unrelated work in immunology, last released
2024-10-18 - and the two differed only in case, which the policy treats as a conflict and
which archiving does not release. Rather than ask for an exception or wait on a name that is
still reserved, the package was renamed.

`rplatypus` appears in neither the current index nor the archive, checked against both. The
name also says what the package is: the computation lives in a Python package, `pyplatypus`,
and this is its R surface - one name per language rather than one name reused.

The GitHub repository was renamed with it, so the URLs in `DESCRIPTION` are the current ones
and the former addresses redirect.

## This package calls Python, and provisions it itself

The computation lives in a Python package, `pyplatypus`, reached through `reticulate`. A
reviewer will reasonably want to know what that does on a machine with no Python and no
network, so:

- **`library(rplatypus)` starts nothing.** The import is wrapped in `reticulate`'s
  `delay_load`, so attaching the package is instant, works offline, and works where no Python
  exists. The engine starts on the first call that needs it.
- **No system Python is touched.** `reticulate::py_require()` declares one exact version of
  `pyplatypus` and `uv` builds an isolated ephemeral environment for it. Nothing is installed
  into a user's Python, and no `PATH` Python is modified or required.
- **Every test that needs the engine skips without it.** The suite is 766 assertions; on a
  machine where the engine cannot start, the ones that need it skip and the rest pass.
  `checking tests ... OK` in the run above was produced that way — the engine never started.
- **The vignettes do not compute.** Each one that trains a model is precomputed from an
  `.Rmd.orig` on the maintainer's machine and ships as a static `.Rmd`, so building the
  vignettes on a CRAN machine renders prose and runs no model. `vignettes/precompute.R`
  documents how they are regenerated.
- **Examples that need the engine are wrapped in `\dontrun{}`**, which was verified by making
  Python unavailable rather than by it happening to be absent: with
  `RETICULATE_PYTHON=/nonexistent/python`, every expression that runs in an example still
  returns - the specification constructors, the colormap and mask helpers, `drawing_style()`,
  `ct_windows()` - and the one function that does need the engine, `available_weights()`, is
  inside `\dontrun{}` already. `checking examples ... OK` on a machine that *has* Python does
  not establish that; this does.

Nothing in `R CMD check` therefore reaches the network or starts Python. The one thing worth
stating plainly because it would surprise a *user* rather than a check: the first real call
downloads a PyTorch wheel of a few gigabytes, which is documented in `?platypus_status` and in
the README, together with how to point the package at an existing install instead.

`SystemRequirements` names Python 3.10 for completeness, though it is not a prerequisite the
user installs.

## Downstream dependencies

None — this is a first submission.
