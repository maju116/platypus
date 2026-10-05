# Build an experiment specification

Either from arguments, or from a YAML file by passing its path as the
only argument. Both produce the same object, and everything downstream
treats them identically.

## Usage

``` r
platypus_spec(
  data,
  models = NULL,
  task = NULL,
  seed = NULL,
  output_dir = NULL,
  check_paths = TRUE
)
```

## Arguments

- data:

  A
  [`segmentation_data()`](https://maju116.github.io/platypus/reference/segmentation_data.md)
  specification, or the path to a YAML file.

- models:

  A list of model specifications, see
  [models](https://maju116.github.io/platypus/reference/models.md).

- task:

  Which task this specification describes, one of
  `"semantic_segmentation"` or `"object_detection"`. Normally left
  unset: the data and model constructors already decide it, and
  [`segmentation_data()`](https://maju116.github.io/platypus/reference/segmentation_data.md)
  with
  [`u_net()`](https://maju116.github.io/platypus/reference/models.md)
  can only mean one thing. Given, it is **checked against them rather
  than trusted**, so it can only agree or refuse. It becomes
  load-bearing the day a pair of constructors stops deciding on its own.

- seed:

  Set it for a reproducible run.

- output_dir:

  Where to write a record of the run: the specification, the history,
  and anything the run worked out that the specification does not
  already say - a detector's fitted anchors, which it cannot be reloaded
  without. `NULL`, the default, writes nothing.

  Unset means *nothing is written*, not "written to a default place". A
  field with a default cannot be told apart from one somebody set to
  that default, so sending the default from here would have put files in
  every user's working directory the moment the engine learned to write
  them - which it just did.

- check_paths:

  Verify that the data paths exist. Turn it off to build a specification
  on a machine that does not hold the data.

## Value

A `platypus_spec`.

## Details

The engine validates it, so a mistake is caught here rather than
part-way through a training run - and reported against the model that
has it, by the name you gave it.

## Examples

``` r
if (FALSE) { # \dontrun{
spec <- platypus_spec(
  data = segmentation_data("train/", "valid/", colormap = binary_colormap),
  models = list(
    u_net("unet", input_shape = c(256, 256)),
    linknet("linknet", input_shape = c(256, 256), loss = loss_focal_tversky(alpha = 0.7))
  )
)

# The same thing, from a file
spec <- platypus_spec("experiment.yaml")
} # }
```
