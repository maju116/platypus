# Read a colour mask back to class indices

The inverse of
[`classes_to_colours()`](https://maju116.github.io/platypus/reference/classes_to_colours.md).
Anything matching no colour becomes background, and `coverage` in the
result says how much of the mask that was - a number near 1 means the
colormap does not describe this dataset, which is the quietest way to
train on nothing.

## Usage

``` r
colours_to_classes(mask, colormap, tolerance = 0)
```

## Arguments

- mask:

  An RGB array: `height x width x 3`, or with a leading image axis.

- colormap:

  A list of RGB triples, one per class.

- tolerance:

  Allow this much difference per channel. Useful after lossy resizing,
  though masks should be resized with `nearest = TRUE` so it is not
  needed.

## Value

A list with `classes` (indices counted from 1) and `coverage` (the
fraction of pixels that matched a colour).

## Examples

``` r
coloured <- classes_to_colours(matrix(c(1, 2, 2, 1), nrow = 2), binary_colormap)
colours_to_classes(coloured, binary_colormap)$classes
#>      [,1] [,2]
#> [1,]    1    2
#> [2,]    2    1
```
