# Read mask files as class indices

Masks come two ways, and this reads both. `colormap` is for masks stored
as pictures, one colour per class. `labels` is for masks stored as label
maps - a vector of the voxel values, in class order - which is how
volume formats store them.

## Usage

``` r
read_masks(paths, colormap = NULL, labels = NULL, size = NULL, tolerance = 0)
```

## Arguments

- paths:

  Image files.

- colormap:

  A list of RGB triples, one per class. Give this or `labels`.

- labels:

  The voxel value of each class, in class order: `c(0, 1)` for a binary
  segmentation. For NIfTI and other label maps. Give this or `colormap`.

- size:

  `c(height, width)` to read at, or `NULL` to keep each image's own
  size - which only works if they all agree.

- tolerance:

  Passed to
  [`mask_classes()`](https://maju116.github.io/platypus/reference/mask_classes.md).
  Ignored for label maps, which are compared to within half a unit - a
  label map that has been resampled or merely passed through a float
  will not satisfy an exact comparison, and a label that fails to match
  becomes background.

## Value

An array of class indices, counted from 1.

## Examples

``` r
if (FALSE) { # \dontrun{
# Masks stored as pictures: the colormap says which colour is which class.
truth <- read_masks(list.files("masks/", full.names = TRUE),
                    colormap = list(c(0, 0, 0), c(255, 255, 255)))

# Masks stored as label maps, which is how volumes do it. Exactly one of the two.
truth <- read_masks("case_01/seg.nii.gz", labels = c(0, 1))

# `tolerance` is for masks that have been through a lossy resize, where a colour
# that should be (255, 0, 0) arrives as (254, 1, 0) and matches nothing.
truth <- read_masks(paths, colormap = voc_colormap, tolerance = 2)
} # }
```
