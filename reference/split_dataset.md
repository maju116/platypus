# Split a dataset into training, validation and test sets

Divides one directory of samples into three sets and writes each as a
CSV the specification can point at. Nothing is copied: the files stay
where they are and the CSVs list them, which matters when the data is
large or read-only.

## Usage

``` r
split_dataset(
  root,
  out_dir,
  group_by = NULL,
  fractions = c(0.7, 0.15, 0.15),
  seed = 0,
  mode = c("nested_dirs", "config_file"),
  subdirs = c("images", "masks"),
  column_sep = ";",
  strict = TRUE,
  relative = TRUE
)
```

## Arguments

- root:

  The directory holding the samples, laid out as `mode` describes.

- out_dir:

  Where to write `train.csv`, `validation.csv` and `test.csv`.

- group_by:

  A pattern picking the group out of a sample's name; see above. `NULL`
  treats every sample as its own group, which is right when one sample
  is one patient.

- fractions:

  Two or three shares that add up to 1: train, validation and optionally
  test. Measured in samples, not in groups, so patients with very
  different numbers of slices do not skew them.

- seed:

  Makes the split reproducible. The same inputs and seed give the same
  split on any machine and in any file order - without that, two runs
  are not comparable and the reason for a difference is impossible to
  find.

- mode:

  `"nested_dirs"` for one directory per sample, `"config_file"` for a
  CSV.

- subdirs:

  The image and mask subdirectory names, for `nested_dirs`.

- column_sep:

  Separator between several paths in one CSV cell.

- strict:

  Treat an incomplete sample as an error rather than skipping it.

- relative:

  Write paths relative to the CSV where possible, so the dataset and its
  split can be moved together.

## Value

A `platypus_split`: the three paths, plus how many samples and groups
each set received. Pass it straight to
[`segmentation_data()`](https://maju116.github.io/platypus/reference/segmentation_data.md).

## Why `group_by` exists

Splitting per file is the obvious thing and, for medical images, the
wrong one. A scan is many slices of one patient, so dividing slices at
random puts the same anatomy in training and in validation at once. The
model then meets the neighbouring slice of what it is being tested on,
validation measures memory rather than generalisation, and Dice comes
out several points too high with nothing in the result to say so.

`group_by` names the unit that must not be split - usually a patient,
sometimes a study, a scanner or a site. Every sample sharing a group
lands in exactly one set.

## Patterns are Python regular expressions

`group_by` is passed to the engine exactly as written, and read there as
a Python regular expression. In practice the dialects agree on
everything you are likely to need

- `^`, `$`, `\\d`, `[A-Z]`, `+`, `*`, `()` - and R's own quoting rules
  still apply, so a backslash is doubled: `"^(patient\\\\d+)_"`. It is
  not translated on the way through, deliberately: a translation that
  quietly got it wrong would produce a working split with the wrong
  groups in it, which is the failure this argument exists to prevent.

A pattern that matches no sample is an error, never a fallback to
one-group-per-sample.

## See also

[`evaluate_cases()`](https://maju116.github.io/platypus/reference/evaluate_cases.md),
which reports a score per case rather than one per set.

## Examples

``` r
if (FALSE) { # \dontrun{
split <- split_dataset(
  "scans", "splits",
  group_by = "^(patient\\\\d+)_",
  fractions = c(0.7, 0.15, 0.15)
)
split

spec <- platypus_spec(
  data = segmentation_data(split, colormap = binary_colormap),
  models = list(u_net("unet", input_shape = c(256, 256)))
)
} # }
```
