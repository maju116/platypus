# Which augmentations are available

The transforms the installed 'albumentations' offers, and - for
volumes - the ones it can actually apply to them.

## Usage

``` r
available_augmentations(rank = 2, pattern = NULL)
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

[`augment()`](https://maju116.github.io/platypus/reference/augment.md),
which uses one.

## Examples

``` r
if (FALSE) { # \dontrun{
available_augmentations()                       # everything, for images
available_augmentations(rank = 3)               # what volumes can take
available_augmentations(rank = 3, pattern = "Flip|Crop")
setdiff(available_augmentations(), available_augmentations(rank = 3))   # the gap
} # }
```
