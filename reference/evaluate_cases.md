# Score every case separately

[`evaluate()`](https://maju116.github.io/platypus/reference/evaluate.md)
gives one row per model: one number for a whole set. This gives one row
per case - each image, or each patient when `group_by` is set - because
one number hides the distribution, and the distribution is usually the
finding. A model averaging Dice 0.86 that scores 0.1 on three patients
has a failure mode, and the mean is where it hides.

## Usage

``` r
evaluate_cases(object, ...)

# S3 method for class 'platypus_fit'
evaluate_cases(
  object,
  model = NULL,
  split = "validation",
  group_by = NULL,
  ...
)

# S3 method for class 'platypus_cases'
summary(object, ...)
```

## Arguments

- object:

  A `platypus_cases` from `evaluate_cases()`.

- ...:

  Unused.

- model:

  Which model, by the name in the specification. Defaults to the first.

- split:

  Which data to score on.

- group_by:

  A pattern picking a group out of each case's name, as in
  [`split_dataset()`](https://maju116.github.io/platypus/reference/split_dataset.md).
  With it, the rows are patients rather than slices: a patient's slices
  are pooled into one score, the way a volume would be.

## Value

A data frame with one row per case, of class `platypus_cases`.

## Details

[`summary()`](https://rdrr.io/r/base/summary.html) on the result gives
the distribution: mean, standard deviation, median, range, and which
case scored worst.

Tiled models are scored on the whole image, not on tiles. Overlaps are
accumulated across a case's tiles and the metric applied once, which is
that image's score exactly; averaging the tiles' scores would be a
different number, and unkind to any case whose object happens to
straddle a boundary.

## See also

[`split_dataset()`](https://maju116.github.io/platypus/reference/split_dataset.md)
for keeping a patient out of two sets in the first place.

## Examples

``` r
if (FALSE) { # \dontrun{
cases <- evaluate_cases(fit)
summary(cases)
head(cases[order(cases$dice), ])          # the ones worth looking at

evaluate_cases(fit, group_by = "^(patient\\\\d+)_")
} # }
```
