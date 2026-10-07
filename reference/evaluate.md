# Compare the trained models

One row per model: what it is, how big it is, how long it trained, and
how it scored.

## Usage

``` r
evaluate(object, ...)

# S3 method for class 'platypus_fit'
evaluate(object, split = "validation", ...)
```

## Arguments

- object:

  A
  [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md).

- ...:

  Unused.

- split:

  Which data to score on: `"validation"`, `"train"` or `"test"`.

## Value

A data frame, one row per model.

## Details

The `loss` column is only comparable between models trained on the same
loss, which is why `loss_function` is there beside it. A Focal-Tversky
of 0.05 is not better than a CCE-Dice of 0.14; it is not even the same
question. Metrics stay comparable, because they measure the mask rather
than the objective.

## A note on the name

`evaluate` is also exported by the **evaluate** package, which knitr
depends on, so the two mask each other depending on which was attached
last. It is a generic here, so `evaluate(fit)` dispatches on the fit's
class whichever one you reach - and `platypus::evaluate(fit)` says so
outright. The name is kept because it is the word for what this does;
[`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md)
carries a prefix only because `fit` is a generic in the tidymodels
packages and a plain `fit` would have been theirs to own.

## Examples

``` r
if (FALSE) { # \dontrun{
evaluate(fit)                      # one row per model, on the validation split
evaluate(fit, split = "test")

# A split with no masks is refused by name rather than scored against nothing;
# `predict()` still works on it.
} # }
```
