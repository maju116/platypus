# Train the models in a specification

Trains every model the specification asks for, in order, each with its
own data pipeline - so two models may differ in input size or tiling and
still be compared fairly, on the same data.

## Usage

``` r
platypus_fit(
  spec,
  device = NULL,
  num_workers = "auto",
  strict_data = TRUE,
  check_masks = TRUE,
  verbose = FALSE
)
```

## Arguments

- spec:

  A
  [`platypus_spec()`](https://maju116.github.io/platypus/reference/platypus_spec.md).

- device:

  `"cuda"`, `"cpu"`, or `NULL` to use a GPU when one is present.

- num_workers:

  Processes used to read and decode images. `"auto"` picks a small
  number based on the machine. This matters more than it sounds: reading
  in the main process alone leaves the GPU waiting on the disk, and on
  the Data Science Bowl at 128x128 that is the difference between 35
  seconds an epoch and under 10. Set `0` to avoid multiprocessing
  entirely if it causes trouble.

- strict_data:

  Treat an incomplete sample as an error. Set `FALSE` to train on the
  rest and be told what was skipped.

- check_masks:

  Read a sample of the training masks before training, and refuse when a
  class the specification declares appears in none of them. A colormap
  or set of labels that matches none of the labelled tissue is the
  quietest way a run wastes a day: every mask reads as background, the
  loss falls because background is most of a medical image, and the
  model learns to answer "nothing here". Set `FALSE` only when the
  missing class is real but rarer than the sample - and see
  [`inspect_masks()`](https://maju116.github.io/platypus/reference/inspect_masks.md)
  first, which asks the same question over as much of the data as you
  like.

- verbose:

  Report each epoch as it finishes.

## Value

A `platypus_fit`.

## Details

Validation data is never augmented. Measuring a model on distorted
images measures the distortion.

This blocks until training finishes. With `verbose = TRUE` each epoch is
reported as it completes, which on anything longer than a few minutes is
the difference between waiting and wondering.

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- platypus_fit(spec, verbose = TRUE)
evaluate(fit)
masks <- predict(fit, "unet")
} # }
```
