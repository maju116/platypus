# Lay a mask over the image it belongs to

The picture that answers "where is it looking". Background - class 1 -
is left alone, so only the classes that were found are tinted.

## Usage

``` r
overlay_mask(image, mask, colormap, alpha = 0.55)
```

## Arguments

- image:

  A `height x width x channels` array. Values in 0-1 or 0-255; greyscale
  is repeated across three channels.

- mask:

  Class indices, counted from 1, matching the image's height and width.

- colormap:

  A list of RGB triples, one per class.

- alpha:

  How strongly to tint, 0 to 1.

## Value

A `height x width x 3` array of 0-255 integers.

## Examples

``` r
image <- array(runif(8 * 8 * 3, 0.2, 0.6), dim = c(8, 8, 3))
mask <- matrix(1L, 8, 8); mask[2:4, 2:4] <- 2L
tinted <- overlay_mask(image, mask, colormap = list(c(0, 0, 0), c(255, 0, 0)))
dim(tinted)
#> [1] 8 8 3
```
