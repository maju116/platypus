# The named CT windows

The windows radiologists use, as `c(centre, width)` in Hounsfield units.
Pass a name to
[`segmentation_data()`](https://maju116.github.io/platypus/reference/segmentation_data.md)
rather than the numbers; this is here for looking them up.

## Usage

``` r
ct_windows()
```

## Value

A named list of `c(centre, width)` pairs.

## Details

Held here rather than fetched from the engine so that looking up a
constant does not require starting Python. A test checks the two lists
against each other whenever the engine is available, so the copy cannot
quietly drift.

## Examples

``` r
ct_windows()$lung
#> centre  width 
#>   -600   1500 
ct_windows()[["soft_tissue"]]
#> centre  width 
#>     40    400 
```
