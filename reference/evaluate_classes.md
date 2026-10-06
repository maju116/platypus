# Average precision per class

The row that matters on unbalanced data, which is most detection data:
BCCD has 4,155 red cells against 372 white and 361 platelets, so a
single number is a number about red cells.
[`evaluate()`](https://maju116.github.io/platypus/reference/evaluate.md)
deliberately carries no overall precision or recall for the same
reason - averaging them over classes needs a weighting, and every choice
of weighting is a different claim.

## Usage

``` r
evaluate_classes(object, ...)

# S3 method for class 'platypus_fit'
evaluate_classes(object, model = NULL, split = "validation", ...)
```

## Arguments

- object:

  A fit from
  [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md)
  on a detection specification.

- ...:

  Unused.

- model:

  Which model, when the specification trained several.

- split:

  Which split to score.

## Value

A data frame, one row per class: average precision at IoU 0.5, the mean
overlap of the boxes that matched, the number of true boxes, and
precision and recall at the model's `operating_point`.

## Examples

``` r
if (FALSE) { # \dontrun{
evaluate_classes(fit, "bccd")

# The row that matters on unbalanced data, which is most data: BCCD has 4155 red
# cells against 372 white and 361 platelets, so a single number is a number about
# red cells.
} # }
```
