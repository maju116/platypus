# Where the images and masks are

Two layouts. `nested_dirs` expects one directory per sample containing
an `images` and a `masks` subdirectory - the Data Science Bowl
arrangement, where a sample may hold dozens of separate mask files that
are combined into one. `config_file` expects a CSV with `images` and
`masks` columns, several paths per cell if needed, resolved relative to
the file itself so a configuration travels with its data.

## Usage

``` r
segmentation_data(
  train,
  validation = NULL,
  colormap = NULL,
  labels = NULL,
  test = NULL,
  split = NULL,
  mode = c("nested_dirs", "config_file"),
  window = NULL,
  dicom_window = NULL,
  target_spacing = NULL,
  channels_from = NULL,
  subdirs = c("images", "masks"),
  column_sep = ";",
  shuffle = TRUE
)
```

## Arguments

- train, validation:

  Paths to the training and validation data. A
  [`platypus_split()`](https://maju116.github.io/platypus/reference/platypus_split.md)
  may be given as `train` on its own: it carries all three paths and
  selects `config_file` mode, so a split needs no unpacking.

- colormap:

  A list of RGB triples, one per class, background first. For masks
  stored as pictures. Give this or `labels`, not both.

- labels:

  The voxel value of each class, in class order, for masks stored as
  label maps - `c(0, 1)` for a binary segmentation, `c(0, 1, 2)` where 1
  is liver and 2 is tumour. This is how NIfTI and every other volume
  format label anything. Give this or `colormap`, not both.

- test:

  Optional test data. Only images are read from it.

- split:

  Divide `train` instead of naming a `validation` set: a list with
  `fractions` (two numbers, or three to cut a test set as well) and
  `group_by`. Exactly one of `split` and `validation`.

  **`group_by` has to be given, even as `NULL`.** It is a regular
  expression read against each sample's name, and everything sharing a
  group lands in a single split - usually a patient, sometimes a study
  or a scanner. `NULL` divides by file instead, and is a perfectly good
  answer when the images are independent.

  It is required rather than optional because the mistake it prevents
  leaves no trace: slices of one patient in training and validation at
  once make validation measure memory rather than generalisation, and
  the score comes out several points too high with nothing in the output
  to say so.

  Nothing is written.
  [`platypus_split()`](https://maju116.github.io/platypus/reference/platypus_split.md)
  is still the way when the three CSV files are the point - to keep, to
  hand to a colleague, to cite - and its result can be passed straight
  to `train`.

- mode:

  `"nested_dirs"` or `"config_file"`.

- window:

  How values in real units reach the model. A named window - `"lung"`,
  `"soft_tissue"`, `"bone"`, `"brain"`, `"abdomen"`, `"liver"`,
  `"mediastinum"`, `"subdural"`, `"stroke"` - or an explicit
  `c(centre, width)` in Hounsfield units, or `"auto"` for the window the
  file recorded, which only DICOM has, or `"full"` for the whole range
  present. Applies to DICOM and to NIfTI; ignored for ordinary pictures,
  which are already 0-255.

  Worth choosing rather than leaving: a fixed window is what makes two
  scans comparable. Scaling each scan by its own darkest and brightest
  voxel lets a single metal implant or marker rescale everything else in
  it.

  `NULL`, the default, sends nothing and lets the engine decide - which
  also keeps ordinary image work running against an engine too old to
  know the setting exists.

- dicom_window:

  The former name of `window`, still accepted. It was accurate while
  DICOM was the only format that needed it.

- target_spacing:

  For volumes: resample every scan to this many millimetres per voxel,
  then centre-crop or pad to the model's `input_shape`. Without it a
  volume is simply resized into that shape, which is only harmless when
  every scan covers the same amount of the patient - and clinical scans
  do not. Forty slices of 1 mm is 40 mm of patient and forty of 2.5 mm
  is 100 mm, so resizing alone leaves the same organ a different size in
  each, and nothing in the data says so.

  `c(1, 1, 1)` is the usual starting point. Ignored for 2D.

- channels_from:

  For datasets that keep one channel per file: one pattern per channel,
  in channel order. BraTS ships four MRI sequences per patient - T1, T1
  after contrast, T2 and FLAIR - because different tumour structures are
  visible in different sequences; satellite sets keep their bands apart
  the same way.

  The order is stated rather than inferred, and that is the point.
  Sorted, the BraTS names come out flair, t1, t1ce, t2 - reproducible
  and anatomically meaningless. A model trained with FLAIR in channel
  one and used on data whose channel one is T1 returns a plausible
  answer with the right shape and the right range, and nothing further
  down can notice.

  Patterns are Python regular expressions, like `group_by` in
  [`platypus_split()`](https://maju116.github.io/platypus/reference/platypus_split.md),
  and R's own quoting applies:
  `c("_t1\\.nii", "_t1ce\\.nii", "_t2\\.nii", "_flair\\.nii")`. Each
  must match exactly one of a sample's files - `"_t1"` would match both
  `_t1.nii.gz` and `_t1ce.nii.gz`, which is an error rather than a
  race - and every file must be claimed by some pattern, because one
  left out is data the model never sees.

- subdirs:

  For `nested_dirs`, the names of the image and mask subdirectories.

- column_sep:

  For `config_file`, what separates several paths in one cell.

- shuffle:

  Shuffle the training data between epochs.

## Value

A data specification, to be passed to
[`platypus_spec()`](https://maju116.github.io/platypus/reference/platypus_spec.md).

## Details

Classes are named one of two ways, and exactly one of them. `colormap`
is for masks stored as pictures: a list of RGB triples where position is
the class index. `labels` is for masks stored as label maps - a vector
of the voxel values, in class order - which is how every volume format
stores them, and how a single-channel PNG can be read as well.

A sample that is missing its masks is an error, not a warning - a
warning in a loop over five hundred directories is a warning nobody
reads.

## Examples

``` r
segmentation_data("train/", "valid/", colormap = binary_colormap)
#> $train_path
#> [1] "train/"
#> 
#> $validation_path
#> [1] "valid/"
#> 
#> $test_path
#> NULL
#> 
#> $split
#> NULL
#> 
#> $mode
#> [1] "nested_dirs"
#> 
#> $colormap
#> $colormap[[1]]
#> [1] 0 0 0
#> 
#> $colormap[[2]]
#> [1] 255 255 255
#> 
#> 
#> $labels
#> NULL
#> 
#> $window
#> NULL
#> 
#> $target_spacing
#> NULL
#> 
#> $channels_from
#> NULL
#> 
#> $subdirs
#> [1] "images" "masks" 
#> 
#> $column_sep
#> [1] ";"
#> 
#> $shuffle
#> [1] TRUE
#> 
```
