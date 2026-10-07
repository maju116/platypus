# The files in one part of a split

Reads the CSV that
[`split_dataset()`](https://maju116.github.io/platypus/reference/split_dataset.md)
wrote and hands back the paths ready to use.

## Usage

``` r
split_path(split, which = c("train", "validation", "test"))
```

## Arguments

- split:

  A
  [`split_dataset()`](https://maju116.github.io/platypus/reference/split_dataset.md).

- which:

  `"train"`, `"validation"` or `"test"`.

## Value

A data frame with `key`, `group`, `images` and `masks`, the paths made
absolute. `images` and `masks` hold one path per sample, or several
separated by the split's separator when a sample has several files.

## Details

The reason this exists rather than
[`read.csv()`](https://rdrr.io/r/utils/read.table.html): the CSVs store
paths **relative to themselves**, so that a dataset and its split can be
moved or mounted elsewhere together. Read directly, those paths do not
open from wherever your session happens to be - which is a trap worth
removing rather than documenting.

## Examples

``` r
if (FALSE) { # \dontrun{
split <- split_dataset("scans", "splits", group_by = "^(patient\\\\d+)_")
validation <- split_path(split, "validation")
images <- read_images(validation$images, size = c(32, 32, 32), channels = 1)
} # }
```
