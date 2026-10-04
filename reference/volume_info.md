# What is in a volume file

Shape and voxel spacing, read through the engine so the numbers are the
ones training would use - including the canonical reorientation, which
changes what "the first axis" means.

## Usage

``` r
volume_info(paths)
```

## Arguments

- paths:

  One or more paths to NIfTI files.

## Value

A data frame with one row per file: `path`, `spacing_1/2/3` in
millimetres, `shape_1/2/3` in voxels, and `voxel_ml`, the volume of one
voxel in millilitres.

## See also

[`mask_volume()`](https://maju116.github.io/platypus/reference/mask_volume.md),
which turns a segmentation into millilitres.

## Examples

``` r
if (FALSE) { # \dontrun{
volume_info("scans/case_01/images/ct.nii.gz")
} # }
```
