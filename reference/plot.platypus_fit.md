# Plot what happened during training

One panel per quantity, one line per model. Useful for the thing a final
number cannot tell you: whether it had stopped improving, or was still
going when it ran out of epochs.

## Usage

``` r
# S3 method for class 'platypus_fit'
plot(x, metrics = NULL, ...)
```

## Arguments

- x:

  A
  [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md).

- metrics:

  Which columns to plot; defaults to the losses and metrics.

- ...:

  Unused.

## Value

A `ggplot`.
