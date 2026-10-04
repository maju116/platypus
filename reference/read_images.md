# Read images as the model sees them

Reads through the engine's own pipeline rather than a separate decoder,
so what you plot is what the model was given. A picture drawn from a
differently resized copy is a picture of something the model never saw,
and the disagreements it appears to show may be nothing but the
resizing.

## Usage

``` r
read_images(paths, size = NULL, channels = 3, nearest = FALSE)
```

## Arguments

- paths:

  Image files.

- size:

  `c(height, width)` to read at, or `NULL` to keep each image's own
  size - which only works if they all agree.

- channels:

  3 for colour, 1 for greyscale.

- nearest:

  Use nearest-neighbour resizing. Required for masks: interpolating one
  invents colours belonging to no class, which then quietly become
  background.

## Value

An `image x height x width x channels` array of 0-255 values.

## Examples

``` r
if (FALSE) { # \dontrun{
images <- read_images(list.files("test/", full.names = TRUE), size = c(256, 256))
} # }
```
