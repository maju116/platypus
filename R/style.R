# The decisions a drawing makes, mirrored from the engine rather than restated.
#
# These five values live in `pyplatypus/style.py` and that module is authoritative. They are
# copied here rather than fetched because **drawing in R needs no Python at all** - measured:
# `plot_masks()`, `plot_boxes()` and `overlay_agreement()` return without touching the
# bridge, so someone with masks in R can draw them without a five-gigabyte torch download.
# A default colour that called the engine would end that, which is a worse trade than a
# mirror.
#
# What makes it a mirror and not a second opinion is
# `tests/testthat/test-drawing-style.R`: it asks the engine for `drawing_style()` and
# asserts every value here equals it, so a change on that side fails CI here. Before this
# existed the two agreed because five divergences had been found by laying the two packages'
# README figures side by side and fixing them by hand, which is a state rather than a
# guarantee.
#
# Deliberately **not** mirrored: the anchor plot draws its cloud of box shapes at alpha 0.55
# while the engine uses 0.35. That is a ggplot aesthetic rather than one of these decisions,
# it happens to share a number with `overlay_alpha`, and folding it in would be this file's
# own mistake in reverse.

# ColorBrewer's Dark2, named once. `box_colours` takes its first two entries rather than
# restating them: the comment used to say the two palettes were one decision while the
# source wrote the same hex codes twice, which is this file's own subject.
platypus_dark2 <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a",
                    "#66a61e", "#e6ab02", "#a6761d", "#666666")

platypus_drawing_style <- list(
  # Found, missed, invented. Three colours rather than one, because a missed lesion and a
  # false alarm cost different things and a single-colour overlay hides which you have.
  # Chosen to stay separable in greyscale, which a figure in a paper may become.
  agreement_colours = c(hit = "#3CDC5A", missed = "#E63C3C", false_alarm = "#F0C83C"),

  # Prediction against truth. ColorBrewer's Dark2, which is colourblind-safe: roughly one
  # man in twelve cannot separate red from green, and a figure nobody can read has failed.
  box_colours = c(prediction = platypus_dark2[[2]], truth = platypus_dark2[[1]]),

  # How much of the image an overlay lets through. 0.55 keeps the tissue legible underneath
  # while the mask still reads as a region, and the point of an overlay is that both are
  # visible at once - otherwise a reader cannot judge whether the boundary is right.
  overlay_alpha = 0.55,

  # A box's label: the class, then its score to two decimals. Two rather than three because
  # the third digit of a detector's confidence is noise at any dataset size here - average
  # precision on BCCD moves by 0.0159 between seeds - and printing it implies a precision
  # the model does not have.
  box_label_format = "%s %.2f",

  # One colour per class, for a figure that separates classes rather than predictions from
  # truth: the whole of ColorBrewer's Dark2, of which `box_colours` above is the first two
  # entries, so the palettes are one decision and not two that happen to overlap. Eight,
  # and then it repeats - a figure with more than eight classes cannot be read by colour
  # whatever palette it uses, and cycling says so where inventing a ninth would not.
  #
  # Until engine 0.8.0a1 this plot used ggplot2's default hue scale, which is not
  # colourblind-safe. That is the reason the engine's palette wins rather than symmetry.
  class_colours = platypus_dark2,

  # Boxes below this are not drawn. Deliberately not the engine's `operating_point`, which
  # sits near zero so average precision can integrate the whole ranking; drawing from that
  # tail fills a frame with boxes the model barely considered. A picture is read by a
  # person, so it gets the threshold a person would want.
  box_min_score = 0.5
)

#' The decisions a drawing makes
#'
#' Every value `plot_masks()`, `plot_boxes()`, `overlay_mask()` and `overlay_agreement()`
#' use when you do not give them one: which colours mean found, missed and invented, one
#' colour per class for [plot_anchors()], how much of the image an overlay lets through, how
#' a box is labelled, and which boxes are worth drawing at all.
#'
#' They belong to the engine - `pyplatypus.style` - and are mirrored here so that drawing
#' needs no Python. A test holds the two together, so the same call draws the same picture
#' from either package.
#'
#' Reach for this when you want to match a figure rather than restate it: passing
#' `colours = drawing_style()$box_colours` to something else keeps one source, and typing
#' the hex codes starts a second.
#'
#' @return A named list: `agreement_colours`, `box_colours`, `class_colours`,
#'   `overlay_alpha`, `box_label_format` and `box_min_score`.
#' @seealso [plot_masks()], [plot_boxes()], [overlay_agreement()]
#' @export
#' @examples
#' drawing_style()$agreement_colours[["missed"]]
#'
#' # The same list the engine returns, which is what keeps the two drawings the same.
#' str(drawing_style())
drawing_style <- function() {
  platypus_drawing_style
}
