# Combine several binary masks into one

The Data Science Bowl stores one file per nucleus; datasets like it are
common. Later masks win where they overlap, so an explicit class beats
the background it sits on.

## Usage

``` r
unite_masks(masks)
```

## Arguments

- masks:

  A list of `height x width` matrices of class indices.

## Value

One `height x width` matrix.
