# The anchors a detector used, and whether they were fitted

A detector cannot be reloaded without its anchors, and when they were
fitted rather than named in the specification this is the only record.
[`save_weights()`](https://maju116.github.io/platypus/reference/save_weights.md)
writes them into the sidecar for the same reason.

## Usage

``` r
detection_anchors(object, model = NULL)
```

## Arguments

- object:

  A fit from
  [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md)
  on a detection specification.

- model:

  Which model, when the specification trained several.

## Value

A list: `anchors` as three groups of pairs, `fitted` saying whether they
were fitted to your data, and when they were, the mean overlap they
achieve and a data frame with one row per anchor.
