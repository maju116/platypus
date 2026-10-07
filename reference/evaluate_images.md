# Score every image separately

**Detection only.** On a segmentation fit this refuses by name and
points at
[`evaluate_cases()`](https://maju116.github.io/platypus/reference/evaluate_cases.md).
The name carries no `detection_` prefix on purpose: this is an S3
generic, so a future task gains a method under the same name rather than
a second spelling of one idea.

## Usage

``` r
evaluate_images(object, ...)

# S3 method for class 'platypus_fit'
evaluate_images(
  object,
  model = NULL,
  split = "validation",
  score_threshold = NULL,
  ...
)
```

## Arguments

- object:

  A
  [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md)
  on a detection specification.

- ...:

  Unused.

- model:

  Which model, by the name in the specification. Defaults to the first.

- split:

  Which data to score on.

- score_threshold:

  The confidence at which the counts are read. Unset, the
  specification's `operating_point` is used, which is what it is for:
  "how many were missed" is undefined over the whole ranking, where
  every box the model dimly considered counts as a prediction.

## Value

A data frame of class `platypus_images`, one row per image: `key`,
`n_truth`, `n_predicted`, `matched`, `missed`, `spurious` and
`mean_matched_iou`.

## Details

[`evaluate()`](https://maju116.github.io/platypus/reference/evaluate.md)
gives one row per model and
[`evaluate_classes()`](https://maju116.github.io/platypus/reference/evaluate_classes.md)
one row per class. This gives one row per image, which is the question
asked next: not how well on average, but **which** pictures it fails on.
A mean over a split says 0.86; it does not say the misses are four
frames where the stain is dark, and that difference is usually something
about the data rather than the model.

Sort by `missed` for the frames it cannot see, and by `mean_matched_iou`
for the ones where it finds everything and places it badly. Those are
different problems - the first is usually the data, the second usually
the anchors.

**There is no average precision here, deliberately.** Average precision
is the area under a precision-recall curve, so it is a property of a
ranking over a whole dataset; computed on one image with three boxes it
moves on a single box's rank and means nothing. What is meaningful for
one picture is counting and overlap, which is what the columns carry.

`mean_matched_iou` is `NA` rather than 0 when nothing matched. An image
where the model found nothing and one where it found badly-placed boxes
are different failures, and a zero would merge them.

## See also

[`evaluate_classes()`](https://maju116.github.io/platypus/reference/evaluate_classes.md)
for one row per class,
[`evaluate_cases()`](https://maju116.github.io/platypus/reference/evaluate_cases.md)
for the segmentation counterpart.

## Examples

``` r
if (FALSE) { # \dontrun{
images <- evaluate_images(fit)
head(images[order(-images$missed), ])          # the frames it cannot see
head(images[order(images$mean_matched_iou), ]) # the ones it places badly
} # }
```
