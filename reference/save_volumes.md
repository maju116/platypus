# Save predicted volumes as NIfTI

Writes each mask as a label-map NIfTI carrying the geometry of the scan
it was predicted from. That is the whole reason this is not
[`save_masks()`](https://maju116.github.io/platypus/reference/save_masks.md)
with a different extension: a mask array without an affine cannot be
laid over its scan by any viewer, any registration tool or any volume
calculation - it either lands in the wrong place or is refused.

## Usage

``` r
save_volumes(masks, dir, reference, names = NULL, suffix = "")
```

## Arguments

- masks:

  Masks of class indices, as
  [`predict()`](https://rdrr.io/r/stats/predict.html) returns.

  A **list** of volumes is what `predict(space = "source")` gives, and
  the form to prefer: each mask is already on the grid of the scan it
  was computed from, so the geometry matches by construction and the
  scans need not be the same size as each other.

  An array is also accepted: one volume as `depth x height x width`, a
  stack as `volume x depth x height x width`, probabilities as
  `volume x depth x height x width x class` which are collapsed to the
  most likely class. A single volume's probabilities have the same rank
  as a stack of masks and are not guessed at - collapse them yourself,
  or keep the volume axis.

- dir:

  Where to write. Created if it does not exist.

- reference:

  The scan each mask was predicted from, in the same order. Its affine
  is what the mask is written with, so this is required rather than
  optional.

- names:

  File names, without extension. Taken from `reference` when unset,
  which works when the scans have distinct filenames. They often do
  not - one directory per case, the same `ct.nii.gz` inside each - and
  writing every mask to one name would silently keep only the last, so
  that case is an error asking for this argument.
  `split_path(split, "validation")$key` is usually the answer.

- suffix:

  Appended to each name, for telling two models' output apart.

## Value

The paths written, invisibly.

## Details

Carrying the affine means the mask has to be on the same grid as the
scan, which is what `predict(space = "source")` is for: it maps each
prediction back out of the model's grid onto the scan's. Without it, a
resampled or resized prediction will be refused here rather than written
somewhere wrong.

## See also

[`save_masks()`](https://maju116.github.io/platypus/reference/save_masks.md)
for 2D masks as pictures.

## Examples

``` r
if (FALSE) { # \dontrun{
validation <- split_path(split, "validation")
masks <- predict(fit, split = "validation", space = "source")
save_volumes(masks, "predictions", reference = validation$images)
} # }
```
