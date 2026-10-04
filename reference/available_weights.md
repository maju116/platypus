# The published weights, and what they are

Weights are named rather than downloaded by hand, and a name is pinned
to one commit of the repository holding it - so `"dsbowl-unet"` means
one set of numbers today and the same set next year. This lists what is
published, with the commit each name resolves to.

## Usage

``` r
available_weights()
```

## Value

A data frame with `name`, `repo`, `filename`, `revision` and
`description`.

## Details

Read the descriptions rather than only the names. A model's limitations
decide whether it answers your question: the nuclei weights are
semantic, so touching nuclei come back as one region and a count taken
from them would be wrong.

## See also

[`u_net()`](https://maju116.github.io/platypus/reference/models.md) and
the other architectures, whose `weights` argument takes a name from
here, a `hf://owner/repo/file.safetensors@commit` reference, or a path
to a local file.

## Examples

``` r
if (FALSE) { # \dontrun{
available_weights()

spec <- platypus_spec(
  data = segmentation_data("images/", "images/", colormap = binary_colormap),
  models = list(u_net("nuclei", input_shape = c(256, 256), blocks = 4, filters = 16,
                      weights = "dsbowl-unet", fit = FALSE))
)
masks <- predict(platypus_fit(spec), split = "validation")
} # }
```
