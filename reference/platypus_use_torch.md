# Ask for a particular PyTorch build

Call this before anything starts the engine - straight after
[`library(platypus)`](https://github.com/maju116/platypus) - and before
the first call that needs Python.

## Usage

``` r
platypus_use_torch(build = c("pascal", "default"))
```

## Arguments

- build:

  `"pascal"` for the last PyTorch built against CUDA 12, or `NULL` for
  the default.

## Value

The requirement string now in force, invisibly.

## Details

There is one reason to: **a GeForce GTX 10-series card, or anything
older**. PyTorch 2.8 and later ship CUDA 13 builds, and CUDA 13 dropped
the Maxwell, Pascal and Volta generations outright. No driver update
brings them back. Without this, such a card is simply not used and
everything runs on the processor, perhaps ten times slower, with nothing
obviously wrong.

On anything from Turing - the RTX 20-series onwards - the default is
correct and this is unnecessary.

## Examples

``` r
if (FALSE) { # \dontrun{
library(platypus)
platypus_use_torch("pascal")   # GTX 10-series and older
} # }
```
