# An augmentation step

Names come from the installed 'albumentations', so whatever it offers is
available and a misspelling is caught when the specification is built,
with a suggestion.

## Usage

``` r
augment(name, ...)
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
augment("HorizontalFlip", p = 0.5)
#> $name
#> [1] "HorizontalFlip"
#> 
#> $params
#> $params$p
#> [1] 0.5
#> 
#> 
```
