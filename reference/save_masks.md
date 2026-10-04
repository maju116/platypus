# Write masks to files

Predictions are only half useful while they are an array in a session.
This writes them as ordinary PNGs that open in anything - ImageJ,
QuPath, a viewer, a colleague's machine - and returns the paths, so it
can sit at the end of a pipeline.

## Usage

``` r
save_masks(masks, dir, colormap, names = NULL, suffix = "")
```

## Arguments

- masks:

  Class indices from
  [`predict.platypus_fit()`](https://maju116.github.io/platypus/reference/predict.platypus_fit.md) -
  an `image x height x width` array, or one `height x width` matrix.
  Probabilities are accepted too and reduced to the most likely class.

  For **volumes** use
  [`save_volumes()`](https://maju116.github.io/platypus/reference/save_volumes.md),
  which writes NIfTI carrying the geometry of the scan each mask came
  from. This function cannot reliably tell a stack of volumes from a
  stack of images with a class axis - the ranks are identical - so it
  does not try; it refuses only the case a shape can prove.

- dir:

  Directory to write into; created if it does not exist.

- colormap:

  A list of RGB triples, one per class.

- names:

  File names, or the source image paths to take names from. Defaults to
  `mask_0001.png` and so on.

- suffix:

  Added before the extension, so predictions from different models can
  sit in one directory without overwriting each other.

## Value

The paths written, invisibly.

## Details

Files go where you say, not beside the images they came from. Writing
into somebody's dataset directory is a surprising thing for a function
to do, and a second run against a different model would quietly mix its
results in with the first.

## Examples

``` r
if (FALSE) { # \dontrun{
masks <- predict(fit, "unet", split = "test")
save_masks(masks, "predictions", binary_colormap, suffix = "_unet")
} # }
```
