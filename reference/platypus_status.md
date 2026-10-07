# Is the engine ready?

Reports whether the Python side has been started, and what it is.
Calling this does not start it - use it to check before a long run, or
when something is wrong.

## Usage

``` r
platypus_status()
```

## Value

A list with the engine version, the interpreter in use and where its
environment lives, or `NULL` fields when Python has not been started
yet.

## Why this page shows no output

What this prints is a property of the machine it runs on, so an example
output here would describe whichever machine built this page rather than
yours. Run it and read your own.

## Examples

``` r
if (FALSE) { # \dontrun{
platypus_status()
} # }
```
