# Metrics

What gets reported. Unlike the losses these are computed on the hard
prediction - the mask you will actually be handed - because a metric
taken on probabilities reads higher than the model deserves.

## Usage

``` r
metric_iou(smooth = 1, include_background = TRUE)

metric_dice(smooth = 1, include_background = TRUE)

metric_tversky(alpha = 0.5, smooth = 1, include_background = TRUE)
```

## Arguments

- smooth:

  Smoothing. Zero is allowed and is the honest choice for a reported
  number: with smoothing, a class absent from an image scores a perfect
  1.

- include_background:

  Average over the background class as well. In medical images the
  background is usually most of the picture, so leaving it in can turn a
  model that found nothing into one that looks respectable. Papers
  report foreground only.

- alpha:

  For Tversky, the weight on false negatives.

## Value

A metric specification.

## Examples

``` r
metric_dice(include_background = FALSE)
#> $name
#> [1] "dice"
#> 
#> $smooth
#> [1] 1
#> 
#> $include_background
#> [1] FALSE
#> 
```
