# Predict masks

Predict masks

## Usage

``` r
# S3 method for class 'platypus_fit'
predict(
  object,
  model = NULL,
  split = "test",
  type = c("class", "probability"),
  space = c("model", "source"),
  ...
)
```

## Arguments

- object:

  A
  [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md).

- model:

  Which model, by the name given in the specification. Defaults to the
  first one trained.

- split:

  Which data to predict on.

- type:

  `"class"` gives the class index per pixel, which is the mask to look
  at; `"probability"` keeps the per-class channel.

- space:

  `"model"` or `"source"`; see above. Note that the two return different
  shapes of thing, an array and a list, because they are different
  things.

- ...:

  Unused.

## Value

With `space = "model"`: for `"class"`, an integer array of
`image x height x width`, classes numbered from 1 to match the colormap;
for `"probability"`, the same with a trailing class axis. Tiled models
return images at their original size, reassembled.

With `space = "source"`: a list of such arrays, one per scan, each on
that scan's grid.

## Which grid the answer is on

`space = "model"` returns everything on the grid the model worked at,
stacked into one array. That is what training saw, and it is the only
form that can be a single array, because a stack needs one shape.

`space = "source"` maps each prediction back onto the grid of the scan
it was computed from - undoing the resampling and the crop - and
therefore returns a **list**, one mask per scan. This is what you want
for anything that has to meet the original data: laying a mask over the
scan in a viewer, writing it beside the scan with
[`save_volumes()`](https://maju116.github.io/platypus/reference/save_volumes.md),
or measuring it in millilitres of the scan's own voxels. Scans differ in
size, so the list is not a limitation being worked around - resizing
them to a common shape is exactly how a mask ends up describing anatomy
it was not computed from.

Two things are lost mapping back, and neither can be recovered.
Interpolation softens a boundary, so the mask returns slightly smoother
than the model drew it. And where reading cropped anatomy away, the mask
comes back padded with background - which means *not examined*, not
*nothing there*. Give the model an `input_shape` that covers the anatomy
if that distinction matters.

## Examples

``` r
if (FALSE) { # \dontrun{
masks <- predict(fit, "unet")                     # an array, on the model's grid
masks <- predict(fit, "unet", space = "source")   # a list, on each scan's own grid

# `space = "source"` returns a list and not an array, which is the API being
# honest: scans differ in size, a stacked array needs one shape, and resizing
# them to match is how a mask ends up describing anatomy it was not computed from.
} # }
```
