# Turn a specification back into plain R data

Useful for looking at what the engine made of your arguments, and for
writing it out.

## Usage

``` r
# S3 method for class 'platypus_spec'
as.list(x, ...)
```

## Arguments

- x:

  A `platypus_spec`.

- ...:

  Unused.

## Value

A nested list.

## Examples

``` r
if (FALSE) { # \dontrun{
str(as.list(spec))

# The same thing a YAML file would hold, which is the point: a configuration file
# and this code describe the same run. `yaml::write_yaml(as.list(spec), "run.yaml")`
# produces a file the Python engine reads directly.
} # }
```
