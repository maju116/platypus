# How much of a volume a class occupies, in millilitres

A voxel count answers nothing on its own: the same tumour is 4000 voxels
on one scanner and 12000 on another, and neither number goes in a
report. Multiplied by the volume of a voxel it becomes millilitres,
which is comparable between scans, between patients and against whatever
the radiologist measured by hand.

## Usage

``` r
mask_volume(mask, spacing, class = 2L)
```

## Arguments

- mask:

  A mask of class indices, as
  [`predict()`](https://rdrr.io/r/stats/predict.html) returns - one
  volume, not a stack.

- spacing:

  Voxel size in millimetres, length 3.
  [`volume_info()`](https://maju116.github.io/platypus/reference/volume_info.md)
  reports it; so does the `spacing_1/2/3` columns of its output.

- class:

  Which class to measure, numbered from 1 as the colormap and `labels`
  are. Defaults to 2, which is the foreground in a binary segmentation.

## Value

The volume in millilitres.

## Examples

``` r
mask <- array(1L, dim = c(10, 10, 4))
mask[3:6, 3:6, 2:3] <- 2L
# 32 voxels of 2 x 2 x 5 mm: 32 * 20 mm^3 = 640 mm^3 = 0.64 ml
mask_volume(mask, spacing = c(2, 2, 5))
#> [1] 0.64
```
