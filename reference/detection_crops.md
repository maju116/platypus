# Cut every detection out of the image it was found in

A detector that knows three classes, followed by a classifier that knows
eighty: find the objects, cut them out, hand the pieces to something
else. [`predict()`](https://rdrr.io/r/stats/predict.html) returns boxes
in each image's own pixels precisely so they can be used on the
photograph they came from, and this is what uses them.

## Usage

``` r
detection_crops(
  object,
  model = NULL,
  split = "test",
  score_threshold = NULL,
  context = 0,
  size = NULL,
  fit = "letterbox",
  ...
)
```

## Arguments

- object:

  A
  [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md)
  from a detection specification.

- model:

  Which model, if the fit holds several.

- split:

  Which split's images to read.

- score_threshold:

  Crop at this confidence instead of the specification's
  `operating_point`.

- context:

  Expand every box by this fraction of its own width and height, per
  side.

- size:

  `c(height, width)` every crop is brought to, or `NULL` to keep each as
  cut. Crops of one image differ in size otherwise, which is why this
  returns a list either way.

- fit:

  `"letterbox"` to preserve the aspect by padding, `"stretch"` to fill
  the frame.

- ...:

  Unused.

## Value

A list with one entry per image, each a list of `key`, `crops` (a list
of `height x width x channels` arrays), `boxes`, `scores`, `labels`,
`names` and `dropped`.

## Details

Cut from the image at its **native size**, not from the square the
network sees. Cropping the letterboxed frame would cut a shrunken object
and then scale it back up: on a 1024-wide photograph fed to a 416 model,
every crop would be two fifths of the size the data allows.

Four more decisions, each of which taken the other way changes what a
downstream model sees without changing anything you would notice:

- **Rounded outward**, so the crop contains the whole box. Rounding to
  nearest loses up to a pixel per side, which on a 26-pixel platelet is
  a twelfth of the object.

- **`context` grows with the box**, not by a number of pixels, so one
  value suits a platelet and a white cell. Zero by default: a detector's
  box is tight by construction, and a classifier trained on whole
  objects does worse on something cut exactly at the edge - but
  expanding silently would put things nobody asked for into every crop.

- **`size` letterboxes rather than stretches.** Resizing a tall box to a
  square changes the aspect of every non-square object, and a classifier
  then sees a shape that does not occur in nature. `fit = "stretch"` is
  there for models trained that way, and has to be asked for.

- **Read at `operating_point`, not at the specification's
  `score_threshold`.** That one sits near zero so that average precision
  integrates the whole ranking - hundreds of boxes an image, almost all
  of them the tail it exists to measure over. Cropping those would hand
  a classifier mostly noise.

`dropped` is the number of boxes that clipped to nothing. A model may
place a box off the frame and
[`predict()`](https://rdrr.io/r/stats/predict.html) reports what it
said, so at a low threshold some arrive with no extent at all; those are
dropped and counted rather than the arrays quietly coming back shorter
than the detections.

## Examples

``` r
if (FALSE) { # \dontrun{
found <- detection_crops(fit, size = c(224, 224))
length(found[[1]]$crops)
plot_masks(found[[1]]$crops[[1]])        # look at one

# Everything above the operating point, as one flat list ready for another model.
pieces <- unlist(lapply(found, `[[`, "crops"), recursive = FALSE)
} # }
```
