# Look at masks beside the images they came from

Builds a grid: one row per image, one column per panel. Which panels
appear depends on what you pass - the image alone, the truth, the
prediction, and where they disagree.

## Usage

``` r
plot_masks(
  images,
  prediction = NULL,
  truth = NULL,
  colormap = binary_colormap,
  which = NULL,
  alpha = 0.55,
  labels = NULL,
  slice = NULL
)
```

## Arguments

- images:

  An `image x height x width x channels` array, or one
  `height x width x channels` image.

- prediction:

  Predicted class indices, from
  [`predict.platypus_fit()`](https://maju116.github.io/platypus/reference/predict.platypus_fit.md).

- truth:

  True class indices, if you have them.

- colormap:

  A list of RGB triples, one per class.

- which:

  Which images to show.

- alpha:

  How strongly to tint the overlays.

- labels:

  Row labels; defaults to the image number.

- slice:

  For volumes, which slice to draw. A volume cannot honestly be shown as
  one picture, so one plane is chosen and said out loud rather than a
  projection being invented. Counted along the last axis, which after
  the canonical reorientation is the axial direction - the view a
  radiologist scrolls through. `"middle"` takes the middle slice, which
  is the sensible first look.

## Value

A `ggplot`.

## Details

The disagreement panel is usually the one worth looking at. It separates
what was found from what was missed and what was invented, and those are
not interchangeable: for most clinical questions a missed lesion and a
false alarm cost different things, and a single overlap score hides
which one you have.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_masks(images, prediction = masks, truth = truth, colormap = binary_colormap)
} # }
```
