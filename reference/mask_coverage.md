# How much of each class a mask covers

A quick sanity check before training: a foreground of half a percent is
a problem the loss has to be chosen for, and it is better to know before
the run than after.

## Usage

``` r
mask_coverage(mask, labels = NULL)
```

## Arguments

- mask:

  Class indices, counted from 1.

- labels:

  Optional names for the classes.

## Value

A data frame with one row per class.

## Examples

``` r
mask <- matrix(1L, 8, 8); mask[2:4, 2:4] <- 2L
mask_coverage(mask, labels = c("background", "nucleus"))
#>   class      label pixels fraction
#> 1     1 background     55 0.859375
#> 2     2    nucleus      9 0.140625

# A class with 0 pixels is the thing to look for: its target channel is zero
# everywhere, so there is no gradient towards it and the model is never shown
# what it is being asked to find.
```
