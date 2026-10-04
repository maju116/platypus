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
