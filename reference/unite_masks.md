# Combine several binary masks into one

The Data Science Bowl stores one file per nucleus; datasets like it are
common. Later masks win where they overlap, so an explicit class beats
the background it sits on.

## Usage

``` r
unite_masks(masks)
```

## Arguments

- masks:

  A list of `height x width` matrices of class indices.

## Value

One `height x width` matrix.

## Examples

``` r
# Data Science Bowl stores one file per nucleus, so a sample's mask arrives as several
# matrices that have to become one. Later masks win where they overlap.
one <- matrix(1L, 8, 8); one[2:4, 2:4] <- 2L
two <- matrix(1L, 8, 8); two[6:7, 5:7] <- 2L
united <- unite_masks(list(one, two))
table(united)
#> united
#>  1  2 
#> 49 15 
```
