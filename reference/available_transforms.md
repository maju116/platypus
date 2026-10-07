# Which augmentations are available

The transforms the installed 'albumentations' offers, and - for
volumes - the ones it can actually apply to them.

## Usage

``` r
available_transforms(rank = 2, pattern = NULL)
```

## Arguments

- rank:

  2 for images, 3 for volumes.

- pattern:

  Optional regular expression to filter the names, for browsing:
  `"Flip"`, `"3D$"`, `"Elastic|Grid"`.

## Value

A character vector of transform names.

## Details

Worth asking rather than guessing. Support for volumes is uneven: most
geometric transforms work and several intensity ones raise from inside
the library, so `rank = 3` returns a shorter list than `rank = 2`. A
transform outside it is refused when the specification is built, by
name, rather than failing an hour into training.

## See also

[`augmentation_step()`](https://maju116.github.io/platypus/reference/augmentation_step.md),
which uses one.

## Examples

``` r
if (FALSE) { # \dontrun{
available_transforms()                       # everything, for images
#> 130 names, with albumentations 2.0.8

available_transforms(rank = 3)               # what volumes can take
#> 97 - the other 33 raise from inside albumentations when handed a volume

available_transforms(rank = 3, pattern = "Flip|Crop")
#> [1] "AtLeastOneBBoxRandomCrop" "CenterCrop"   "CenterCrop3D"
#> [4] "CropAndPad"   "CropNonEmptyMaskIfExists"  "HorizontalFlip"
#> [7] "RandomCrop"   "RandomCrop3D"              "RandomCropFromBorders"
#> [10] "RandomResizedCrop"  "RandomSizedBBoxSafeCrop"  "RandomSizedCrop"
#> [13] "VerticalFlip"

length(setdiff(available_transforms(), available_transforms(rank = 3)))
#> [1] 33
} # }
```
