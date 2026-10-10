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
