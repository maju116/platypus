# Check folders of DICOM slices before training on them

One row per directory, saying whether it is a usable series and what is
wrong with it if it is not. Problems are reported as a column rather
than raised, because a hundred cases out of an archive will contain a
few with a missing slice, two series in one folder, or duplicated
files - and stopping at the first turns an afternoon's work into a
week's.

## Usage

``` r
series_report(paths)
```

## Arguments

- paths:

  Directories, each holding the slices of one series.

## Value

A data frame with one row per directory: `path`, `ok`, `slices`,
`sorted_by`, `spacing_1/2/3`, `series_uid` and `problem`.

## Details

Reads no pixels. Every check runs on the DICOM headers, so this is
usable on a hundred gigabytes of data and is the first thing worth
running on a new dataset.

## What it catches

A **missing slice**, which does not make a volume shorter - it puts
everything past the gap in the wrong place, and no metric would show it.
**Two series in one folder**, which is the normal state of an archive
export: a scout, a reconstruction and a phase sitting together, which
stacked would interleave two anatomies. **Duplicated slices**, from a
copy or from several echoes mixed together. Slices of different
**size**, **angle** or **spacing**.

It also reports how the slices were ordered. `"position"` means the
geometry decided, which is what you want. `"instance_number"` means the
files carry no positions and the order came from a number that is only
required to be unique - worth knowing before trusting the result.

## See also

[`volume_info()`](https://maju116.github.io/platypus/reference/volume_info.md)
for NIfTI files,
[`split_dataset()`](https://maju116.github.io/platypus/reference/split_dataset.md)
for dividing the cases up.

## Examples

``` r
if (FALSE) { # \dontrun{
cases <- list.dirs("dicom_export", recursive = FALSE)
report <- series_report(cases)

report[!report$ok, c("path", "problem")]     # the ones to deal with first
table(report$sorted_by)
} # }
```
