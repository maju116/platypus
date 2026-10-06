# Average the weights over the last part of the run

Stochastic weight averaging. Past `start`, every epoch's weights are
folded into a running average and at the end that average becomes the
model - the claim being that a point in the middle of a flat region
generalises better than whichever corner the last epoch stopped in.

## Usage

``` r
callback_swa(start = NULL, learning_rate = NULL)
```

## Arguments

- start:

  The fraction of the run after which averaging begins - 0.75 folds the
  last quarter. Unset uses the engine's default of 0.75. A fraction
  rather than an epoch so it survives a change to `epochs`, and the
  final epoch is always folded, so 1 averages one set of weights rather
  than none.

- learning_rate:

  Hold the rate at this value once averaging begins. Unset leaves
  whatever the run was doing.

  Worth setting, and the reason is the whole mechanism: averaging is
  only worth something while the weights are still moving, and a rate
  that has decayed towards zero produces a set of nearly identical
  snapshots whose average is the last one. For the same reason a run
  cannot ask for this and
  [`callback_cosine_annealing()`](https://maju116.github.io/platypus/reference/callbacks.md)
  at once - the two undo each other and the specification refuses the
  pair.

## Value

A callback specification, to be passed to a model constructor.

## Details

**What it is worth is not what a table of scores shows.** Three seeds,
60 epochs, lesions with ambiguous edges:

|                  |                |                |                  |
|------------------|----------------|----------------|------------------|
| run              | Dice           | volume bias    | volume \|error\| |
| plain            | 0.9639 ±0.0100 | +1.71% ±12.18% | 12.60% ±4.19%    |
| `callback_swa()` | 0.9667 ±0.0087 | +2.08% ±2.15%  | 12.38% ±4.31%    |

Dice does not move and neither does the per-case error - both are inside
the seed noise. What moves is **reproducibility**: the plain runs'
volume bias was +4.5%, +12.2% and -11.6% across three seeds, and with
averaging -0.1%, +4.2% and +2.1%. A spread 5.7 times tighter, improving
on all three seeds when paired, by 7.34% at 4.9 standard errors.

So read it as *"the same model twice"* rather than *"a better model"* -
which is what averaging is for, and the reason it belongs beside
[`mask_volume()`](https://maju116.github.io/platypus/reference/mask_volume.md)
in any argument about whether a measured volume can be compared across
runs.

**Batch-normalisation statistics are recomputed afterwards**, by a pass
over the training data. They have to be: an averaged weight tensor
inherits the statistics of whichever epoch was last rather than
averaging them, so without the pass the model is evaluated under the
wrong normalisation and scores far worse than it should with nothing to
say why. Models here have batch normalisation on by default, so this is
nearly every run.

## See also

[callbacks](https://maju116.github.io/platypus/reference/callbacks.md)
for the others,
[`mask_volume()`](https://maju116.github.io/platypus/reference/mask_volume.md)
for reading a volume in millilitres.

## Examples

``` r
callback_swa()
#> $name
#> [1] "swa"
#> 
callback_swa(start = 0.5, learning_rate = 1e-3)
#> $name
#> [1] "swa"
#> 
#> $start
#> [1] 0.5
#> 
#> $learning_rate
#> [1] 0.001
#> 
```
