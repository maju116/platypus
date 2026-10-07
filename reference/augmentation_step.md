# An augmentation step

Names come from the installed 'albumentations', so whatever it offers is
available and a misspelling is caught when the specification is built,
with a suggestion.

## Usage

``` r
augmentation_step(name, ...)
```

## Arguments

- name:

  The transform, for instance `"HorizontalFlip"`.

- ...:

  Its parameters, for instance `p = 0.5`.

## Value

An augmentation specification.

## Examples

``` r
augmentation_step("HorizontalFlip", p = 0.5)
#> $name
#> [1] "HorizontalFlip"
#> 
#> $params
#> $params$p
#> [1] 0.5
#> 
#> 
```
