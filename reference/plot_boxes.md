# Draw boxes on images

The detection counterpart of
[`plot_masks()`](https://maju116.github.io/platypus/reference/plot_masks.md),
and the function the 2020 package had under this name. Boxes arrive from
[`predict()`](https://rdrr.io/r/stats/predict.html) in each image's own
pixels, so they land on the picture without anything being undone first.

## Usage

``` r
plot_boxes(
  images,
  boxes,
  truth = NULL,
  which = NULL,
  min_score = 0.5,
  labels = NULL,
  colours = c(prediction = "#d95f02", truth = "#1b9e77"),
  size = 0.6,
  text_size = 3
)
```

## Arguments

- images:

  An image stack from
  [`read_images()`](https://maju116.github.io/platypus/reference/read_images.md),
  or a single image.

- boxes:

  Boxes to draw: one data frame as
  [`predict()`](https://rdrr.io/r/stats/predict.html) returns per image,
  or a list of them - one per image in `images`. Columns `xmin`, `ymin`,
  `xmax`, `ymax`, and optionally `name` and `score`.

- truth:

  Boxes to draw as ground truth, the same shape as `boxes`.

- which:

  Which images to draw. Defaults to the first four, like
  [`plot_masks()`](https://maju116.github.io/platypus/reference/plot_masks.md).

- min_score:

  Drop predicted boxes below this confidence before drawing. The default
  follows a detector's own `score_threshold` of 0.01, which is kept low
  so average precision can be computed over the whole ranking - and
  which puts far more boxes on a picture than anyone wants to look at.
  `0.5` is the usual choice for a figure.

- labels:

  Row labels. Defaults to the names of `boxes`, which
  [`predict()`](https://rdrr.io/r/stats/predict.html) sets to the sample
  keys.

- colours:

  Named vector giving the colour for `prediction` and for `truth`.

- size:

  Line width of a box.

- text_size:

  Size of the class label, or `NULL` to draw no labels.

## Value

A ggplot object.

## Details

Predictions and truth can be drawn together, and when they are, the
point is the comparison: truth in one colour, predictions in another,
both on the same image. A detector that found the right number of
objects in the wrong places and one that found the wrong number in the
right places score similarly and look nothing alike.

## See also

[`predict.platypus_fit()`](https://maju116.github.io/platypus/reference/predict.platypus_fit.md),
[`plot_masks()`](https://maju116.github.io/platypus/reference/plot_masks.md)

## Examples

``` r
if (FALSE) { # \dontrun{
found <- predict(fit, split = "test")
plot_boxes(read_images(files, size = NULL), found, min_score = 0.5)
} # }
```
