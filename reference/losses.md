# Loss functions

The objective the model is trained against. Every one of these works
unchanged on volumes as well as images.

## Usage

``` r
loss_iou(smooth = 1)

loss_dice(smooth = 1)

loss_cce(label_smoothing = 0)

loss_cce_dice(cce_weight = 0.5, smooth = 1)

loss_focal(gamma = 2, alpha = NULL)

loss_tversky(alpha = 0.5, smooth = 1)

loss_focal_tversky(alpha = 0.5, gamma = 1, smooth = 1)

loss_combo(alpha = 0.5, ce_ratio = 0.5)

loss_lovasz(per_image = FALSE)
```

## Arguments

- smooth:

  Added to numerator and denominator to keep an empty class finite.

- label_smoothing, cce_weight, ce_ratio, per_image:

  See the individual losses.

- gamma:

  For focal losses, how hard to discount the pixels already classified
  well.

- alpha:

  For Tversky, the weight on false negatives; raising it buys recall.
  Note that `alpha = 0.5` is Dice exactly only when `smooth = 0`.

## Value

A loss specification, to be passed to a model constructor.

## Details

There is a tenth, documented separately because it needs a page of its
own:
[`loss_boundary()`](https://maju116.github.io/platypus/reference/loss_boundary.md)
adds the signed distance to the truth's boundary on top of any of these,
which removes the systematic volume bias at some cost in per-case
accuracy.

## Examples

``` r
loss_focal(gamma = 2)
#> $name
#> [1] "focal"
#> 
#> $gamma
#> [1] 2
#> 
#> $alpha
#> NULL
#> 
loss_tversky(alpha = 0.7)
#> $name
#> [1] "tversky"
#> 
#> $alpha
#> [1] 0.7
#> 
#> $smooth
#> [1] 1
#> 
```
