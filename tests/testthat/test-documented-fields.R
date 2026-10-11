# Every configuration field the engine describes has to reach an R reader too.
#
# `CLAUDE.md` §3C used to say a new field needed "nothing to write", because the field's
# `description=` is the documentation "on both sites". Only the engine generates a page from
# the schema; R has none. An R reader meets a field through `?platypus_spec` or a
# constructor's `@param`, both written by hand - so a field described only in the schema is
# undocumented for half the users, and nothing said so.
#
# This is the cheap half of platypus#135: presence, not agreement. It catches a field added
# to the schema and documented nowhere in R. It would *not* have caught `seed`, which was
# present on both sides and stale on one for several releases - that needs the two texts
# held to each other, which is a design question rather than a test, and is still open.

#' Fields the engine describes, flattened across every definition in the schema
described_fields <- function() {
  defs <- platypus:::engine()$spec_schema()$`$defs`
  found <- character(0)
  for (definition in defs) {
    properties <- definition$properties
    if (!length(properties)) next
    for (name in names(properties)) {
      if (nzchar(properties[[name]]$description %||% "")) found <- c(found, name)
    }
  }
  unique(found)
}

# Absent from R's documentation on purpose, each for its own reason. A name earns a place
# here by being unreachable from R rather than by being inconvenient.
deliberately_absent <- c(
  # The constructor *is* the architecture in R: `u_net()` cannot mean anything else, and a
  # field naming it would be a second way to say the same thing (§3A).
  architecture = "the constructor's own name in R",
  # Arrives through `augmentation_step(name, ...)`, so the arguments are the transform's and
  # documented by albumentations rather than here.
  params = "passed through `...` to the named transform",
  # R keeps a family of its own shape, per §3A: `segmentation_data(train, validation, test)`.
  train_path = "`train` on the data constructors",
  validation_path = "`validation` on the data constructors",
  test_path = "`test` on the data constructors"
)

test_that("every configuration field the engine describes is documented in R", {
  skip_if_no_engine()
  documented <- documented_arguments()  # nolint: object_usage_linter.
  skip_if(!length(documented), "no Rd database reachable from here")

  missing <- setdiff(described_fields(), c(documented, names(deliberately_absent)))
  expect_identical(
    missing, character(0),
    info = paste(
      "described in the schema and documented nowhere in R:", paste(missing, collapse = ", "),
      "\nEither write the `@param`, or add the name to `deliberately_absent` with its reason."
    )
  )
})

test_that("nothing on the deliberately-absent list is actually documented", {
  # An allow-list that permits something already present is a hole rather than a list: the
  # day `train_path` gains an `@param`, this list would keep excusing the next field to go
  # missing under that name. §4's "a check calibrated loosely enough admits the failure it
  # was written to prevent", in the one place such a list can rot.
  skip_if_no_engine()
  documented <- documented_arguments()  # nolint: object_usage_linter.
  skip_if(!length(documented), "no Rd database reachable from here")

  stale <- intersect(names(deliberately_absent), documented)
  expect_identical(
    stale, character(0),
    info = paste("documented after all, so remove from `deliberately_absent`:",
                 paste(stale, collapse = ", "))
  )
})

# The other half of platypus#135: agreement, not presence.
#
# `tests/testthat/fixtures/field-prose.yml` records, per field, the digest of the engine's
# own description at the moment somebody read it and wrote or checked R's `@param`. The
# presence test above cannot notice a description being rewritten; this one fails the moment
# it is, and names the help topics to re-read. `tools/record-field-prose.R` writes the file.

test_that("the ledger covers exactly the fields the engine describes", {
  skip_if_no_engine()
  skip_if_not_installed("yaml")
  recorded <- recorded_field_prose()
  skip_if(is.null(recorded), "fixtures/field-prose.yml not reachable from here")

  described <- described_field_prose()
  expect_gt(length(described), 80)        # or this passes by comparing two empty sets

  listed <- vapply(recorded$fields, function(entry) entry$field, character(1))
  expect_identical(
    sort(setdiff(names(described), listed)), character(0),
    info = "described by the engine and absent from the ledger: run tools/record-field-prose.R"
  )
  expect_identical(
    sort(setdiff(listed, names(described))), character(0),
    info = "in the ledger and no longer described by the engine: run tools/record-field-prose.R"
  )
})

test_that("every recorded digest still matches the engine's own text", {
  skip_if_no_engine()
  skip_if_not_installed("yaml")
  recorded <- recorded_field_prose()
  skip_if(is.null(recorded), "fixtures/field-prose.yml not reachable from here")

  described <- described_field_prose()
  ledger <- stats::setNames(
    vapply(recorded$fields, function(entry) entry$engine_digest, character(1)),
    vapply(recorded$fields, function(entry) entry$field, character(1))
  )
  topics <- lapply(
    stats::setNames(recorded$fields, vapply(recorded$fields, function(e) e$field, character(1))),
    function(entry) entry$documented_in
  )

  shared <- intersect(names(described), names(ledger))
  moved <- shared[described[shared] != ledger[shared]]
  expect_identical(
    moved, character(0),
    info = paste0(
      "the engine rewrote the description of: ",
      paste(sprintf("%s (documented in %s)", moved,
                    vapply(moved, function(f) paste(topics[[f]], collapse = ", "),
                           character(1))),
            collapse = "; "),
      ".\nRe-read R's prose for each, then run tools/record-field-prose.R. ",
      "Regenerating without reading is how a ledger becomes a rubber stamp."
    )
  )
})

test_that("the ledger points at help topics that really document the field", {
  skip_if_no_engine()
  skip_if_not_installed("yaml")
  recorded <- recorded_field_prose()
  skip_if(is.null(recorded), "fixtures/field-prose.yml not reachable from here")
  locations <- argument_locations()
  skip_if(!length(locations), "no Rd database reachable from here")

  wrong <- character(0)
  for (entry in recorded$fields) {
    claimed <- entry$documented_in
    real <- locations[[entry$field]]
    # An empty claim is the case the presence test above owns, and `deliberately_absent`
    # explains the five that have one. Here only a *wrong* pointer is a failure.
    for (topic in claimed) {
      if (!topic %in% real) {
        wrong <- c(wrong, sprintf("%s -> %s", entry$field, topic))
      }
    }
  }
  expect_identical(
    wrong, character(0),
    info = paste("the ledger sends a reader to a topic that does not document the field:",
                 paste(wrong, collapse = "; "))
  )
})

test_that("the ledger was written against the engine this package pins", {
  skip_if_no_engine()
  skip_if_not_installed("yaml")
  recorded <- recorded_field_prose()
  skip_if(is.null(recorded), "fixtures/field-prose.yml not reachable from here")

  # Without this, a ledger regenerated against some other engine on somebody's machine would
  # read as agreement with the pin. The digests would all match each other and none of them
  # would be about the engine R actually runs.
  expect_identical(as.character(recorded$engine), platypus_status()$engine_version)
})

# And the third thing platypus#135 asked for: not presence, not agreement, but substance.
#
# The failure this catches is a stub standing in for a described field. It happened twice and
# at scale: `@param learning_rate,weight_decay Standard.` and
# `@param beta_1,beta_2,eps,amsgrad,momentum,nesterov,alpha,rho,lr_decay Per optimiser.`
# documented eleven fields with two phrases, against 1,900 characters of engine text.
#
# **The threshold comes from a measured gap, not from taste.** Over the 88 fields documented
# on both sides, R's text against the engine's longest for that name:
#
#     before the prose pass of 2026-10-11   median 0.68, 22 fields below 0.25
#     after it                              median 0.99, thinnest 0.23 (`monitor`)
#     the stubs it replaced                 0.04 ("Standard."), 0.08 ("Per optimiser.")
#
# So 0.15 sits in the gap between the thinnest real text and the fattest stub. A ratio is a
# crude instrument - platypus#135 says so - and this is the one job it can do without
# nagging: a floor at the median would flag half the package for no reason.

test_that("no field the engine describes is documented in R with a stub", {
  skip_if_no_engine()
  lengths <- prose_lengths()
  skip_if(!length(lengths$r), "no Rd database reachable from here")
  expect_gt(length(lengths$r), 80)

  ratio <- lengths$r / lengths$engine
  stubs <- sort(ratio[ratio < 0.15])
  expect_identical(
    names(stubs), character(0),
    info = paste0(
      "these say far less in R than the engine says about them: ",
      paste(sprintf("%s (%d chars against %d)", names(stubs),
                    lengths$r[names(stubs)], lengths$engine[names(stubs)]),
            collapse = "; "),
      ".\nWrite the `@param`, in R's own words - the engine's text is the source, not the ",
      "thing to copy."
    )
  )
})

test_that("the measure is on the Rd a reader sees, not on the roxygen", {
  # Same reason `documented_arguments()` reads the Rd: a roxygen comment that never
  # regenerated would pass a grep over R/ and still be absent from `?optimizers`. This
  # asserts the instrument reads something, so the test above cannot pass on an empty read.
  skip_if_no_engine()
  lengths <- prose_lengths()
  skip_if(!length(lengths$r), "no Rd database reachable from here")
  expect_gt(lengths$r[["learning_rate"]], 100)
  expect_gt(lengths$r[["eps"]], 60)
  expect_gt(lengths$engine[["seed"]], 500)
})
