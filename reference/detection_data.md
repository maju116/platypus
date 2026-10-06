# Where the images and their annotation files are

The detection counterpart of
[`segmentation_data()`](https://maju116.github.io/platypus/reference/segmentation_data.md).
Two layouts, the same two as segmentation: `nested_dirs` expects one
directory per sample holding an `images` and an `annotations`
subdirectory, and `config_file` expects a CSV with `images` and
`annotations` columns. A detection dataset in Pascal VOC's own layout -
one flat `JPEGImages` beside one flat `Annotations` - is neither, and
the way in is to write three CSV files listing the pairs;
`examples/detect_blood_cells.py` in the engine does exactly that for
BCCD in a dozen lines.

## Usage

``` r
detection_data(
  train,
  validation = NULL,
  classes,
  test = NULL,
  split = NULL,
  mode = c("nested_dirs", "config_file"),
  annotation_format = c("pascal_voc", "labelme"),
  coordinates = NULL,
  strict_labels = TRUE,
  window = NULL,
  subdirs = c("images", "annotations"),
  column_sep = ";",
  shuffle = TRUE
)
```

## Arguments

- train, validation:

  Paths to the training and validation data.

- classes:

  Class names, **in class order**. Position is the class index the model
  learns, and that is why it is written down rather than read from the
  files: sorted, BCCD's three come out `Platelets, RBC, WBC`, which is
  reproducible and anatomically meaningless. A model trained with `RBC`
  first and used where the first class is `WBC` reports plausible boxes
  with plausible scores and nothing downstream can tell.

  There is no background entry, unlike a segmentation colormap. A pixel
  always belongs to some class, so segmentation needs a name for "none
  of them"; a region of an image holding no object is simply not a box.
  One class is therefore a legitimate thing to ask for.

- test:

  Optional test data. Annotations are read if they are there, and the
  split is then scoreable; without them it can be predicted and
  [`evaluate()`](https://maju116.github.io/platypus/reference/evaluate.md)
  refuses it by name, because scoring against empty truth would report
  zero and a zero in a table reads as a result.

- split:

  Divide `train` instead of naming a `validation` set: a list with
  `fractions` (two numbers, or three to cut a test set too) and
  `group_by`, which is required and may be `NULL`. Exactly one of
  `split` and `validation`.

  This is the only way to divide one folder for detection.
  [`platypus_split()`](https://maju116.github.io/platypus/reference/platypus_split.md)
  writes a `masks` column and so cannot be used here; `split` divides
  the samples themselves, so column names never come into it. A test set
  cut this way carries its annotations, so unlike a separate `test`
  folder it can be scored and not only predicted on.

- mode:

  `"nested_dirs"` or `"config_file"`.

- annotation_format:

  `"pascal_voc"` for the XML most detection datasets ship, or
  `"labelme"` for its JSON.

- coordinates:

  How to read a Pascal VOC box. The format's own convention is 1-based
  and inclusive of both ends, so a ten-pixel-wide box runs
  `xmin=1 xmax=10` and the width is `xmax - xmin + 1`. Plenty of tools
  write the same files 0-based and exclusive, where that box is
  `xmin=0 xmax=10`. Reading one as the other shifts every box by a pixel
  and shrinks it by one, which is invisible at a glance and costs real
  overlap on a small object - at 24 pixels a side, one pixel is 4%.

  `NULL`, the default, means the VOC convention. Only meaningful for
  `pascal_voc`: LabelMe stores continuous pixel coordinates and has no
  convention to pick, so giving both is an error rather than a setting
  that does nothing.

- strict_labels:

  Refuse an object whose class is not in `classes`. `FALSE` skips it
  instead, which is how you train on two of a dataset's three classes.
  `TRUE` by default, because an unexpected label is more often a typo in
  `classes` than an intention.

- window:

  How values in real units reach the model, for DICOM and NIfTI. See
  [`segmentation_data()`](https://maju116.github.io/platypus/reference/segmentation_data.md);
  ignored for ordinary pictures, which is what most detection data is.

- subdirs:

  For `nested_dirs`, the names of the image and annotation
  subdirectories.

- column_sep:

  For `config_file`, what separates several paths in one cell.

- shuffle:

  Shuffle the training data between epochs.

## Value

A data specification, to be passed to
[`platypus_spec()`](https://maju116.github.io/platypus/reference/platypus_spec.md).

## See also

[`yolo3()`](https://maju116.github.io/platypus/reference/yolo3.md),
[`plot_boxes()`](https://maju116.github.io/platypus/reference/plot_boxes.md)

## Examples

``` r
detection_data("train/", "valid/", classes = c("RBC", "WBC", "Platelets"))
#> $train_path
#> [1] "train/"
#> 
#> $validation_path
#> [1] "valid/"
#> 
#> $validation
#> NULL
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
#> $classes
#> [1] "RBC"       "WBC"       "Platelets"
#> 
#> $annotation_format
#> [1] "pascal_voc"
#> 
#> $coordinates
#> NULL
#> 
#> $strict_labels
#> [1] TRUE
#> 
#> $window
#> NULL
#> 
#> $subdirs
#> [1] "images"      "annotations"
#> 
#> $column_sep
#> [1] ";"
#> 
#> $shuffle
#> [1] TRUE
#> 
#> attr(,"class")
#> [1] "platypus_detection_data" "list"                   
#> attr(,"task")
#> [1] "object_detection"
```
