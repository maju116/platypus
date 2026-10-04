# Draw the anchors against the boxes they were fitted to

The picture the 2020 package drew, and the one no summary statistic
replaces. Every annotated box is a point - its width against its height,
both as fractions of the model's input - coloured by class, with the
anchors on top.

## Usage

``` r
plot_anchors(object, model = NULL, split = "train", log = FALSE, size = 1.1)
```

## Arguments

- object:

  A fit from
  [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md)
  on a detection specification.

- model:

  Which model, when the specification trained several.

- split:

  Which split's boxes to draw.

- log:

  Draw both axes on a log scale. Detection datasets span a wide range of
  sizes

  - a BCCD platelet is about two fifths the side of a red cell and a
    fifth of a white one - and on linear axes the smallest class
    collapses into the corner.

- size:

  Point size for the boxes.

## Value

A ggplot object.

## Details

What it answers that a mean overlap does not: **whether a class has any
anchor near it at all**. A mean of 0.65 can be one class covered well
and another not covered at all, and those two situations want different
fixes. On blood cells the three classes occupy three distinct regions -
median sides of 133, 69 and 26 pixels at a 416 input - and anchors
fitted to all three together cover them at 0.88, 0.85 and 0.83, so
fitting to the mixture does not abandon the smallest class. COCO's nine
cover the same three at 0.64, 0.70 and 0.70: *evenly* worse rather than
blind to one, which is the useful thing to know. They are not aimed
elsewhere - they span 10 to 373 pixels a side because COCO holds objects
of every size, and most of that range describes nothing here.

Worth drawing for a split the anchors were **not** fitted on. Anchors
that sit among the training boxes and away from the validation ones say
the two halves hold different objects, and no training curve shows that.

## See also

[`detection_anchors()`](https://maju116.github.io/platypus/reference/detection_anchors.md)
for the numbers,
[`yolo3()`](https://maju116.github.io/platypus/reference/yolo3.md) for
choosing them.

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- platypus_fit(spec)
plot_anchors(fit)                      # the split they were fitted to
plot_anchors(fit, split = "validation")  # and one they were not
} # }
```
