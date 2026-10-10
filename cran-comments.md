# cran-comments

First submission of `platypus`.

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

The check reports a conflict with `Platypus`, archived from CRAN on 2026-02-02. The two
differ only in case, which the policy treats as a conflict, and archiving does not release a
name.

This R package has been called `platypus` since 2020 and is published under that name on
GitHub, with a Python engine of its own on PyPI. The archived package is unrelated work in
immunology, last released 2024-10-18.

I have written to its former maintainer and to CRAN about reusing the name, and I am happy to
be guided. **If the name cannot be used I will resubmit under another one** rather than ask for
an exception.

## This package calls Python, and provisions it itself

The computation lives in a Python package, `pyplatypus`, reached through `reticulate`. A
reviewer will reasonably want to know what that does on a machine with no Python and no
network, so:

- **`library(platypus)` starts nothing.** The import is wrapped in `reticulate`'s
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
