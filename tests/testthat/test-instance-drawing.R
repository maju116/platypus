# The colouring rule, held to the engine's rather than to a reading of it.
#
# `#154`'s own words: "the colouring rule has to be the same rule, not the same intention",
# and "two packages could colour instances differently while every shared constant agrees".
# That is exactly the hole, because the palette is mirrored and the mirror test would keep
# passing while the rule diverged - adjacency against identity, four-connected against
# eight, a different tie-break in the greedy.
#
# So these draw the same picture both ways and compare it **pixel for pixel**. Nothing
# weaker can hold a rule: a test that both pictures have three colours in them passes on two
# different colourings, and this project has written that kind of test before.

draw_in_engine <- function(what, arguments) {
  reticulate::py_run_string(sprintf(
    "import numpy as np, pyplatypus\nout = pyplatypus.%s(%s)", what, arguments
  ))$out
}

# Deliberately not a round number of objects and not a tidy grid: the colouring is decided
# by the adjacency graph, so the pictures worth comparing are the ones whose graph is not
# trivial. Case 3 is the one that forces a third colour - three objects each touching the
# other two - which is where a greedy with a different tie-break first disagrees.
instance_pictures <- function() {
  seam <- matrix(0L, nrow = 6, ncol = 10)
  seam[2:5, 2:4] <- 1L
  seam[2:5, 5:7] <- 2L

  corner <- matrix(0L, nrow = 6, ncol = 6)
  corner[2:3, 2:3] <- 1L
  corner[4:5, 4:5] <- 2L   # touching only at a corner, which is eight-connected

  triangle <- matrix(0L, nrow = 7, ncol = 7)
  triangle[2:4, 2:4] <- 1L
  triangle[2:4, 5:6] <- 2L
  triangle[5:6, 2:6] <- 3L  # each of the three touches the other two

  apart <- matrix(0L, nrow = 6, ncol = 10)
  apart[2:3, 2:3] <- 1L
  apart[2:3, 7:8] <- 2L     # no adjacency at all, so both may reuse slot 0

  chain <- matrix(0L, nrow = 5, ncol = 12)
  for (index in seq_len(5)) {
    chain[2:4, (2 * index):(2 * index + 1)] <- index
  }

  # A gap in the numbering, which is what a decoding with `minimum_area` leaves behind: it
  # drops the small objects and does not renumber, so this is the ordinary shape of a
  # prediction rather than a contrived one. The two packages agree on it.
  #
  # It is here after a mutation went looking for a hole and found that there is none:
  # taking `seq_len(max(labels))` instead of the values actually present changes nothing,
  # because a phantom object has degree 0, so it sorts last in Welsh-Powell and cannot take
  # a slot from a real neighbour - and its own blend touches no pixels. Recorded because the
  # conclusion is "equivalent code", not "untested code", and the next person to try that
  # mutation should not have to work it out twice.
  gapped <- matrix(0L, nrow = 7, ncol = 9)
  gapped[2:4, 2:4] <- 1L
  gapped[2:4, 5:6] <- 3L
  gapped[5:6, 2:6] <- 7L

  list(seam = seam, corner = corner, triangle = triangle, apart = apart, chain = chain,
       gapped = gapped)
}

test_that("the instance colouring is the engine's colouring, pixel for pixel", {
  skip_if_no_engine()
  for (name in names(instance_pictures())) {
    labels <- instance_pictures()[[name]]
    image <- array(0.5, dim = c(dim(labels), 3L))
    reticulate::py_run_string(sprintf(
      "import numpy as np\nlabels = np.array(%s, dtype=int).reshape(%d, %d, order='F')
image = np.full((%d, %d, 3), 0.5)",
      paste0("[", paste(as.integer(labels), collapse = ","), "]"),
      nrow(labels), ncol(labels), nrow(labels), ncol(labels)
    ))
    theirs <- draw_in_engine("overlay_instances", "image, labels")
    mine <- overlay_instances(image, labels)
    expect_identical(dim(mine), dim(theirs), info = name)
    expect_equal(as.vector(mine), as.vector(theirs), info = name)
  }
})

test_that("the outline is the engine's outline, pixel for pixel", {
  skip_if_no_engine()
  # A blob with a staircase edge and a real interior, not one of the pictures above: those
  # objects are three pixels across, so every pixel of them is already on the boundary and
  # an outline rule could be wrong in any way without the picture changing. Measured - a
  # mutation adding a diagonal to the interior test passed against `triangle` and fails
  # against this. That is §4's "a check calibrated loosely enough admits the failure it was
  # written to prevent", found by mutating rather than by reading.
  labels <- matrix(0L, nrow = 11, ncol = 13)
  labels[3:8, 3:8] <- 1L
  labels[5:8, 5:8] <- 0L        # an L, so the object turns a corner
  labels[3:6, 10:12] <- 2L      # and a solid neighbour, which has an interior
  image <- array(0.5, dim = c(dim(labels), 3L))
  reticulate::py_run_string(sprintf(
    "import numpy as np\nlabels = np.array(%s, dtype=int).reshape(%d, %d, order='F')
image = np.full((%d, %d, 3), 0.5)",
    paste0("[", paste(as.integer(labels), collapse = ","), "]"),
    nrow(labels), ncol(labels), nrow(labels), ncol(labels)
  ))
  theirs <- draw_in_engine("overlay_instances", "image, labels, outline=True")
  expect_equal(as.vector(overlay_instances(image, labels, outline = TRUE)),
               as.vector(theirs))
})

test_that("the object matching is the engine's matching, pixel for pixel", {
  skip_if_no_engine()
  # Five cases, chosen as the ways an instance result goes wrong rather than as shapes: one
  # exact, one merged pair, one split object, one invented, one missed. A merged pair and a
  # split object are the two that are nearly perfect pixelwise and wrong, which is the whole
  # reason this picture is drawn in objects.
  truth <- matrix(0L, nrow = 7, ncol = 14)
  truth[2:5, 2:4] <- 1L
  truth[2:5, 6:8] <- 2L
  truth[2:5, 10:12] <- 3L

  merged <- truth
  merged[2:5, 2:8] <- 1L
  split <- truth
  split[4:5, 2:4] <- 4L
  invented <- truth
  invented[6, 2:4] <- 5L
  missed <- truth
  missed[missed == 3L] <- 0L
  cases <- list(exact = truth, merged = merged, split = split,
                invented = invented, missed = missed)

  image <- array(0.5, dim = c(dim(truth), 3L))
  as_python <- function(name, m) {
    sprintf("%s = np.array(%s, dtype=int).reshape(%d, %d, order='F')", name,
            paste0("[", paste(as.integer(m), collapse = ","), "]"), nrow(m), ncol(m))
  }
  for (name in names(cases)) {
    prediction <- cases[[name]]
    reticulate::py_run_string(sprintf(
      "import numpy as np\n%s\n%s\nimage = np.full((%d, %d, 3), 0.5)",
      as_python("truth", truth), as_python("prediction", prediction),
      nrow(truth), ncol(truth)
    ))
    theirs <- draw_in_engine("overlay_instance_agreement", "image, prediction, truth")
    mine <- overlay_instance_agreement(image, prediction, truth)
    expect_equal(as.vector(mine), as.vector(theirs), info = name)
  }
})

test_that("a threshold below 0.5 is refused on both sides, for the same reason", {
  expect_error(
    overlay_instance_agreement(array(0.5, dim = c(4, 4, 3)),
                               matrix(0L, 4, 4), matrix(0L, 4, 4), threshold = 0.4),
    "no longer one to one"
  )
  skip_if_no_engine()
  # The engine refuses it too, and the test says so rather than trusting the two messages to
  # have been written from the same understanding.
  refused <- reticulate::py_run_string("
import numpy as np, pyplatypus
try:
    pyplatypus.overlay_instance_agreement(np.full((4, 4, 3), 0.5), np.zeros((4, 4), int),
                                          np.zeros((4, 4), int), threshold=0.4)
    out = ''
except pyplatypus.PlatypusError as error:
    out = str(error)
")$out
  expect_match(refused, "no longer one to one")
})

test_that("a one-hot array is refused, because a channel cannot carry an identity", {
  expect_error(
    overlay_instances(array(0.5, dim = c(4, 4, 3)), array(0L, dim = c(4, 4, 2))),
    "one integer per pixel"
  )
})

test_that("an empty palette is refused rather than drawing nothing", {
  labels <- matrix(0L, 4, 4)
  labels[2, 2] <- 1L
  expect_error(
    overlay_instances(array(0.5, dim = c(4, 4, 3)), labels, colours = character()),
    "nothing to draw the objects in"
  )
})

test_that("a colouring never gives two touching objects one colour", {
  # The property itself, asserted on the pictures above rather than only against the engine:
  # if both packages were wrong in the same way the comparison tests would pass.
  for (name in names(instance_pictures())) {
    labels <- instance_pictures()[[name]]
    slots <- rplatypus:::instance_colour_index(labels, 8L)
    pairs <- rplatypus:::instance_adjacent_pairs(labels)
    for (row in seq_len(nrow(pairs))) {
      expect_false(
        identical(slots[[as.character(pairs[row, 1])]],
                  slots[[as.character(pairs[row, 2])]]),
        info = paste(name, pairs[row, 1], pairs[row, 2])
      )
    }
  }
})

test_that("objects that touch nothing may share a colour, which is the point", {
  # Adjacency rather than identity. A test that demanded distinct colours per object would
  # pass on a rule that needs a ninth colour for a ninth nucleus.
  slots <- rplatypus:::instance_colour_index(instance_pictures()$apart, 8L)
  expect_identical(unname(unlist(slots)), c(0L, 0L))
})

