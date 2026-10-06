# YOLOv3

One detector, as an entry in
[`platypus_spec()`](https://maju116.github.io/platypus/reference/platypus_spec.md)'s
`models`.

## Usage

``` r
yolo3(
  name,
  input_shape = c(416, 416),
  channels = NULL,
  anchors = NULL,
  anchors_per_grid = NULL,
  ignore_threshold = NULL,
  score_threshold = NULL,
  nms_threshold = NULL,
  operating_point = NULL,
  min_visibility = NULL,
  box_loss = NULL,
  optimizer = NULL,
  callbacks = NULL,
  augmentation = NULL,
  epochs = NULL,
  batch_size = NULL,
  weights = NULL,
  fit = NULL
)
```

## Arguments

- name:

  Unique within a spec; names the outputs.

- input_shape:

  `c(height, width)`, each a multiple of 32. The three grids are the
  input divided by 32, 16 and 8, so a side that does not divide gives a
  coarsest grid that does not tile the image. Bigger is not obviously
  better: on BCCD, 608 bought 0.014 of mAP@0.5 for twice the training
  time, and the grid does not quantise a box's centre - the strides are
  fixed and the offset is a continuous output, so a larger input buys
  more cells rather than finer ones.

- channels:

  Channels the model takes.

- anchors:

  Box shapes to predict offsets from, **as fractions of the input
  size**: a list of three groups, coarsest grid first, each group a list
  of `c(width, height)` pairs. COCO publishes its nine in pixels at a
  416 input, so those need dividing by 416. `NULL` fits them to your
  data.

- anchors_per_grid:

  How many anchors to fit per grid when `anchors` is unset. It also sets
  the head's width, which is `anchors_per_grid * (n_class + 5)`.

- ignore_threshold:

  A cell that is not responsible for an object but whose box overlaps
  one this much is left unsupervised rather than trained towards "no
  object". Without it, a nearly-correct guess beside the responsible
  cell is punished, which is the one thing a detector should not learn.

- score_threshold:

  Discard a predicted box below this confidence. Low on purpose: average
  precision is computed over the whole ranking, so a cut here is a
  ceiling on recall that no threshold chosen later can lift.

- nms_threshold:

  Two boxes of the same class overlapping this much are one object and
  the lower-scoring one is dropped. Per class, so a platelet on a red
  cell survives.

- operating_point:

  The confidence at which precision and recall are reported. Separate
  from `score_threshold` because they answer different questions: that
  one decides what is computed at all, this one decides where on the
  curve to stand.

- min_visibility:

  How much of a box must survive an augmentation that removes part of
  the frame, as a fraction of its original area. A convention rather
  than a measurement; what is defensible is the direction. A box keeping
  two pixels of a cell teaches the model that a two-pixel fragment is a
  whole cell, which produces false positives everywhere, while dropping
  a heavily truncated object only fails to teach it about that object.
  Irrelevant unless a transform can lose part of the frame, which flips
  and rotations never do.

- box_loss:

  How the box coordinates are scored: `"offsets"`, YOLOv3's own, or
  `"giou"`. Unset uses the engine's default, which is `"offsets"` - what
  the published `bccd-yolo3` weights were trained with.

  **`"giou"` is not the better objective, and that is measured.** Blood
  cells, three seeds each, 150 epochs, on the dataset's own test split:

  |            |                |                 |                    |
  |------------|----------------|-----------------|--------------------|
  | `box_loss` | mAP@0.5        | mAP@\[.50:.95\] | matched IoU        |
  | `giou`     | 0.8666 ±0.0147 | 0.5078 ±0.0083  | 0.7929 ±0.0069     |
  | `offsets`  | 0.8583 ±0.0167 | 0.5172 ±0.0195  | **0.8038 ±0.0001** |

  Average precision separates them in neither direction - both gaps are
  smaller than the seed spread. The overlap of the boxes it matched does
  separate them, and `"giou"` is worse by 0.011, which is 2.7 standard
  errors of the difference. It is also sixty times less repeatable on
  exactly that quantity.

  The reason is visible in
  [`detection_anchors()`](https://maju116.github.io/platypus/reference/detection_anchors.md):
  GIoU exists because plain IoU is a flat zero for boxes that do not
  touch, so it has no gradient where a detector is most wrong - and
  anchors fitted to blood cells already cover them at a mean IoU of
  0.877, so a prediction never starts disjoint and the advantage never
  arrives. On data whose fitted anchors cover poorly it should be a
  different story, which is untested and so not claimed.

  **Choose it to read the loss, not to raise the score.** The offsets
  term cannot reach zero - cross-entropy against a soft target bottoms
  out at that target's entropy - so a converged run and a stalled one
  print the same number, near 5.9. Under `"giou"` the coordinate term in
  [`training_history()`](https://maju116.github.io/platypus/reference/training_history.md)
  reads about 0.05 on training against 0.47 on validation: a
  localisation gap you can see, because zero means the boxes are right.

- optimizer, callbacks, augmentation, epochs, batch_size:

  As for
  [`u_net()`](https://maju116.github.io/platypus/reference/models.md).
  There is no `loss` and no `metrics`: YOLOv3's objective is part of its
  architecture, and mean average precision is not one option among
  several. A field that accepts a value and ignores it is worse than no
  field.

- weights:

  A registry name such as `"bccd-yolo3"`, an
  `hf://owner/repo/file@commit` reference, or a path to a local file.
  The anchors come with it.

- fit:

  Set `FALSE` to load weights and skip training, which is how you get
  boxes on an image without a GPU and an afternoon.

## Value

One model specification.

## Details

**Anchors.** Unset, they are fitted to your training boxes with k-means
under an IoU distance, which is what you want: COCO's nine anchors
borrowed for blood cells cover their boxes at a mean overlap of 0.65
against 0.88 for anchors fitted to them. Fitted anchors are recorded
with the run and in the sidecar when weights are exported, because **a
detector cannot be reloaded without them** - the same weights read with
other anchors decode every box scaled by a fixed factor, with plausible
boxes, plausible scores, wrong places, and nothing in the output to say
so.

For the same reason, naming `weights` and `anchors` together is an
error: loading takes the anchors recorded beside the file, which would
leave the ones here describing nothing.

## See also

[`detection_data()`](https://maju116.github.io/platypus/reference/detection_data.md),
[`plot_boxes()`](https://maju116.github.io/platypus/reference/plot_boxes.md),
[`available_weights()`](https://maju116.github.io/platypus/reference/available_weights.md)

## Examples

``` r
yolo3("cells", input_shape = c(416, 416))
#> $name
#> [1] "cells"
#> 
#> $architecture
#> [1] "yolo3"
#> 
#> $input_shape
#> [1] 416 416
#> 
#> attr(,"class")
#> [1] "platypus_detection_model" "list"                    
#> attr(,"task")
#> [1] "object_detection"
yolo3("cells", input_shape = c(416, 416), weights = "bccd-yolo3", fit = FALSE)
#> $name
#> [1] "cells"
#> 
#> $architecture
#> [1] "yolo3"
#> 
#> $input_shape
#> [1] 416 416
#> 
#> $weights
#> [1] "bccd-yolo3"
#> 
#> $fit
#> [1] FALSE
#> 
#> attr(,"class")
#> [1] "platypus_detection_model" "list"                    
#> attr(,"task")
#> [1] "object_detection"
```
