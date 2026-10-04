# Paint a mask with its colormap

Paint a mask with its colormap

## Usage

``` r
mask_colours(mask, colormap)
```

## Arguments

- mask:

  Class indices, counted from 1: a `height x width` matrix, or an
  `image x height x width` array for several at once.

- colormap:

  A list of RGB triples, one per class; see
  [binary_colormap](https://maju116.github.io/platypus/reference/colormaps.md).

## Value

The same shape with a trailing RGB axis, values 0-255.

## Examples

``` r
mask <- matrix(c(1, 1, 2, 2), nrow = 2)
dim(mask_colours(mask, binary_colormap))
#> [1] 2 2 3
```
