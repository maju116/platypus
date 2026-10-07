# Callbacks

Things that happen between epochs. `monitor` is checked against the
metrics the model actually asks for, so a callback watching a number
that will never arrive is refused when the specification is built rather
than waited on forever.

## Usage

``` r
callback_early_stopping(
  monitor = "val_loss",
  patience = 10,
  min_delta = 0,
  restore_best = TRUE
)

callback_model_checkpoint(path, monitor = "val_loss", save_best_only = TRUE)

callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.1,
  patience = 5,
  min_lr = 0
)

callback_cosine_annealing(min_lr = 0, epochs = NULL)

callback_csv_logger(path)

callback_terminate_on_nan()
```

## Arguments

- monitor:

  What to watch: `"val_loss"`, `"train_loss"`, or `"val_"` followed by a
  metric name, such as `"val_dice"`.

- patience:

  Epochs to wait before acting.

- min_delta:

  Improvement smaller than this does not count.

- restore_best:

  Put the best weights back when training stops.

- path:

  Where to write.

- save_best_only:

  Only write when the watched quantity improves.

- factor, min_lr:

  For the learning rate schedule. `min_lr` is the floor for both
  `callback_reduce_lr_on_plateau()` and `callback_cosine_annealing()`.

- epochs:

  For `callback_cosine_annealing()`, how many epochs to spread the decay
  over. Unset means the model's `epochs`.

## Value

A callback specification.

## Details

Whether the watched quantity should rise or fall is worked out from its
name: anything ending in `loss` is minimised, everything else maximised.

There is a seventh, documented separately because it needs a page of its
own:
[`callback_swa()`](https://maju116.github.io/platypus/reference/callback_swa.md)
averages the weights over the last part of the run. It watches nothing
and changes how reproducible a run is rather than how well it scores.

## Cosine annealing

`callback_cosine_annealing()` decays the learning rate from its initial
value to `min_lr` along a cosine. It watches nothing: it is a function
of how far through the run you are rather than of how the run is going,
which is the difference from `callback_reduce_lr_on_plateau()` - and a
run may legitimately want both.

Each parameter group decays from its **own** initial rate, so an
`encoder_learning_rate` set on a model is not flattened at the first
epoch. That would have been invisible: the history records one
`learning_rate`, the first group's.

`epochs` unset means the model's own `epochs`, which is what you want -
a cosine that ends where the run ends.

## Examples

``` r
callback_early_stopping(monitor = "val_dice", patience = 10)
#> $name
#> [1] "early_stopping"
#> 
#> $monitor
#> [1] "val_dice"
#> 
#> $patience
#> [1] 10
#> 
#> $min_delta
#> [1] 0
#> 
#> $restore_best
#> [1] TRUE
#> 
```
