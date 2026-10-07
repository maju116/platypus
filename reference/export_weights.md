# Save a trained model's weights

Writes safetensors plus a `.json` sidecar recording what the weights are
for: architecture, input shape, channels, classes, and whatever else you
pass. Loading refuses a model they do not belong to, which is what the
sidecar is for - weights trained on a different colormap with the same
number of classes load cleanly and predict nonsense.

## Usage

``` r
export_weights(object, path, model = NULL, ...)
```

## Arguments

- object:

  A
  [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md).

- path:

  Where to write, with or without the `.safetensors` extension.

- model:

  Which model, by the name in the specification. Defaults to the first.

- ...:

  Extra fields for the sidecar, for instance `data = "BBBC038v1"`,
  `licence = "CC0 1.0"`.

## Value

The path written, invisibly, with the sidecar's contents attached as the
`recorded` attribute - so you can see what was put beside the weights
without opening the file.

## Details

Anything in `...` joins the sidecar. The data they were trained on and
its licence belong there: weights whose provenance lives in somebody's
memory cannot be used by anybody else, including you in a year.

## Examples

``` r
if (FALSE) { # \dontrun{
export_weights(fit, "weights/my-run",
             data = "our 2026 cohort", licence = "internal use only")
} # }
```
