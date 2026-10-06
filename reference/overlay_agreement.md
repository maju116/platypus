# Where a prediction and the truth disagree

Three colours rather than one: what was found, what was missed, and what
was invented. A Dice of 0.85 says nothing about which of the three is
costing you, and for most clinical questions they are not
interchangeable - a missed lesion and a false alarm are different kinds
of wrong.

## Usage

``` r
overlay_agreement(
  image,
  prediction,
  truth,
  alpha = 0.55,
  colours = c(hit = "#3CDC5A", missed = "#E63C3C", false_alarm = "#F0C83C")
)
```

## Arguments

- image:

  A `height x width x channels` array.

- prediction, truth:

  Class indices, counted from 1.

- alpha:

  How strongly to tint.

- colours:

  Named colours for `hit`, `missed` and `false_alarm`.

## Value

A `height x width x 3` array of 0-255 integers.

## Examples

``` r
image <- array(runif(8 * 8 * 3, 0.2, 0.6), dim = c(8, 8, 3))
truth <- matrix(1L, 8, 8); truth[2:4, 2:4] <- 2L
predicted <- matrix(1L, 8, 8); predicted[3:5, 2:4] <- 2L
shown <- overlay_agreement(image, prediction = predicted, truth = truth)
dim(shown)
#> [1] 8 8 3

# Three colours, not one: a missed lesion and a false alarm cost different things,
# and a single overlap score hides which one you have.
```
