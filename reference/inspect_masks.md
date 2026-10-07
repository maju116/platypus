# What the colormap or labels match in your masks

Reads a sample of one split's masks and reports what the specification's
colours, or list of labels, actually matched. Trains nothing and loads
no model.

## Usage

``` r
inspect_masks(spec, split = c("train", "validation", "test"), limit = 500)
```

## Arguments

- spec:

  A
  [`platypus_spec()`](https://maju116.github.io/platypus/reference/platypus_spec.md).

- split:

  `"train"`, `"validation"` or `"test"`.

- limit:

  How many masks to read. They are spread across the split rather than
  taken from its start, because datasets arrive sorted and a prefix
  would answer about the beginning.

## Value

A one-row data frame: `unmatched`, `samples_checked`, `total_samples`,
and `present_classes` and `missing_classes` as comma-separated strings
so the frame stays printable. The classes are also attached as integer
vectors in the attributes `present` and `missing`.

## Details

[`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md)
does a small version of this before every run and refuses when a
declared class appears in none of the masks it read. This is the same
question asked deliberately, over as much of the data as you want -
which is what to reach for when that refusal looks wrong.

## What the numbers mean

`missing_classes` is the one to act on. A class that appears in no mask
cannot be learned: its channel of the target is zero everywhere, so
there is no gradient towards it and the model is never shown the thing
it is being asked to find. Either the colours do not describe these
masks, or they name a class the data does not contain.

`unmatched` is the fraction of mask pixels matching no entry, which fall
back to the background class. **Read it knowing that it scales with the
size of the thing being segmented**, so it is loud for a large structure
and almost silent for a small lesion. Measured on masks that are white,
with a colormap asking for a colour that is not there: a foreground
covering 20% of the image gives 19.8%, and one covering 0.6% gives
0.61% - which is less than the 0.72% that JPEG compression leaves around
the edge of a perfectly correct mask. That is why the refusal is built
on class presence instead.

## Examples

``` r
if (FALSE) { # \dontrun{
inspect_masks(spec)
inspect_masks(spec, split = "validation", limit = 2000)
} # }
```
