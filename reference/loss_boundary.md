# A loss that knows how far a wrong voxel is from the truth

Dice and IoU count a voxel the same wherever it sits, which is why a
model can reach 0.88 on Dice while its volumes run a fifth too large:
the overshoot is all at the boundary, and for a small lesion the
boundary is most of the object. This adds `mean(phi * p)`, where `phi`
is the signed distance to the truth's boundary - negative inside,
positive outside - so a voxel predicted far outside the truth costs in
proportion to how far.

## Usage

``` r
loss_boundary(region = NULL, alpha = NULL)
```

## Arguments

- region:

  The overlap term this is added to - any other loss. Unset uses the
  engine's default, which is Dice.
  [`loss_focal()`](https://maju116.github.io/platypus/reference/losses.md)
  here is what this feature was originally asked for: focal combined
  with a distance metric.

- alpha:

  How much of the objective is the region term, `1 - alpha` the surface
  term. Unset uses the engine's default of 0.9.

  **Not 0.5, and that was measured.** At 0.5 the surface term owns half
  the objective while the prediction is still random, which is where it
  is unstable: three seeds gave a volume error spread of ±28.28% against
  the baseline's ±3.11%, with every number worse. Left unset here on
  purpose - the number is a measurement and belongs in the one place the
  measurement lives, so that an engine that learns better cannot be
  contradicted from this side.

## Value

A loss specification, to be passed to a model constructor.

## Details

**What it buys and what it costs, measured.** Three seeds, 60 epochs,
lesions with ambiguous edges, against Dice alone:

|  |  |  |  |
|----|----|----|----|
| loss | Dice | volume bias | volume \|error\| |
| [`loss_dice()`](https://maju116.github.io/platypus/reference/losses.md) | 0.9525 ±0.0027 | −5.18% ±3.11% | 16.90% ±0.90% |
| `loss_boundary()` | 0.9411 ±0.0093 | −0.06% ±4.15% | 22.17% ±2.88% |

It trades **accuracy per case** for **being unbiased over a series**.
"Is there a lesion" wants the overlap; "has it grown since March" wants
a volume that is not systematically wrong, and no single number answers
both - which is the same point
[`mask_volume()`](https://maju116.github.io/platypus/reference/mask_volume.md)
makes about reporting millilitres beside an overlap score. The bias
improvement is suggestive on three seeds, since the baseline's own bias
wanders ±3.11%; the costs are established.

**It costs time too.** The signed distance transform runs once per
sample, in the loader's workers: 5.3 ms for a 256x256 image, 23.4 ms for
a 64x64x32 volume and 709.8 ms at 128^3. On large volumes give
[`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md)
workers, or the transform becomes the training.

## See also

[losses](https://maju116.github.io/platypus/reference/losses.md) for the
nine region losses,
[`mask_volume()`](https://maju116.github.io/platypus/reference/mask_volume.md)
for reading a volume in millilitres rather than voxels.

## Examples

``` r
loss_boundary()
#> $name
#> [1] "boundary"
#> 
#> $region
#> NULL
#> 
#> $alpha
#> NULL
#> 
loss_boundary(region = loss_focal(gamma = 2))
#> $name
#> [1] "boundary"
#> 
#> $region
#> $region$name
#> [1] "focal"
#> 
#> $region$gamma
#> [1] 2
#> 
#> $region$alpha
#> NULL
#> 
#> 
#> $alpha
#> NULL
#> 
loss_boundary(region = loss_dice(), alpha = 0.95)
#> $name
#> [1] "boundary"
#> 
#> $region
#> $region$name
#> [1] "dice"
#> 
#> $region$smooth
#> [1] 1
#> 
#> 
#> $alpha
#> [1] 0.95
#> 
```
