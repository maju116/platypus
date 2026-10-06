# The epoch-by-epoch record

The epoch-by-epoch record

## Usage

``` r
training_history(object)
```

## Arguments

- object:

  A
  [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md).

## Value

A data frame with one row per model per epoch.

## Examples

``` r
if (FALSE) { # \dontrun{
history <- training_history(fit)
head(history)

# For a detector the loss is broken into four terms on purpose: a run whose
# coordinate loss falls while its objectness loss does not is finding the right
# places and refusing to commit, and the reverse is confident nonsense.
} # }
```
