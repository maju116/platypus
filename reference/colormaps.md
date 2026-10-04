# Colormaps and their labels

`binary_colormap` is black background, white foreground: the ordinary
two-class problem. `voc_colormap` is the twenty-one class palette used
by Pascal VOC, kept because datasets and other people's masks still
arrive in it.

## Usage

``` r
binary_colormap

binary_labels

voc_colormap

voc_labels
```

## Format

A list of RGB triples, or a character vector of names.

## Details

The labels are the matching class names, for figure legends and
[`mask_coverage()`](https://maju116.github.io/platypus/reference/mask_coverage.md).
Position in the list is the class index throughout, counting from 1.

## Examples

``` r
mask_coverage(matrix(c(1, 2, 2, 1), 2), labels = binary_labels)
#>   class      label pixels fraction
#> 1     1 background      2      0.5
#> 2     2     object      2      0.5
```
