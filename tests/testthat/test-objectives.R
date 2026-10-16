# The two objectives the pin moved to 0.8.0a26 for, and the questions R owns about them.
#
# R forwards a loss and a metric as a list of fields; it cannot evaluate either, so the
# arithmetic is the engine's suite's job and is proven there against `scipy.ndimage` rather
# than against a second transcription of the same formula. What R owns is three things the
# engine cannot check from its side: that the name and the fields R's constructors send are
# ones the engine accepts, that the help pages do not say something the engine disagrees
# with, and that a tolerance which cannot mean anything is refused at the line it was
# written on rather than when a run starts.
#
# Everything below offers the engine a configuration with `check_paths = FALSE`, which is
# the shape the encoder and regions probes use: it asks about the specification and nothing
# about this disk.
#
# Comparing prose across the two packages is only safe because of the pin - one exact
# engine, so a rewording on either side fails here instead of drifting - which is
# `CLAUDE.md` §3F's argument for the five pairs already in that table.

semantic_config <- function(..., model = list()) {
  list(
    task = "semantic_segmentation",
    data = list(train_path = ".", validation_path = ".",
                colormap = list(c(0L, 0L, 0L), c(255L, 255L, 255L))),
    models = list(utils::modifyList(
      list(name = "m", input_shape = c(64L, 64L)), model
    )),
    ...
  )
}

built <- function(config) {
  rplatypus:::shim()$build_spec(config, check_paths = FALSE)
}

rd_text <- function(topic) {
  for (root in c(".", "..", file.path("..", ".."))) {
    path <- file.path(root, "man", paste0(topic, ".Rd"))
    if (file.exists(path)) return(paste(readLines(path, warn = FALSE), collapse = "\n"))
  }
  db <- tryCatch(tools::Rd_db("rplatypus"), error = function(e) list())
  key <- paste0(topic, ".Rd")
  if (!key %in% names(db)) return(NA_character_)
  paste(unlist(lapply(db[[key]], as.character)), collapse = "\n")
}

test_that("the engine accepts both of them as R's constructors write them", {
  # The join, which neither suite proves on its own: a name in a union and the fields a
  # constructor sends are different claims, and §4's "a feature is not shipped until
  # something has used it the way a user reaches it" is about exactly this gap.
  skip_if_no_engine()

  result <- built(semantic_config(model = list(
    loss = loss_generalized_dice(smooth = 0.5),
    metrics = list(metric_dice(), metric_surface_dice(tau = 1.5))
  )))
  expect_true(isTRUE(result$ok), info = paste(unlist(result$message), collapse = " "))

  # And the fields arrive rather than merely being accepted: an engine that silently
  # dropped `tau` would pass the line above.
  model <- rplatypus:::shim()$spec_as_dict(result$spec)$models[[1]]
  expect_identical(model$loss$name, "generalized_dice")
  expect_equal(model$loss$smooth, 0.5)
  names <- vapply(model$metrics, function(m) m$name, character(1))
  expect_true("surface_dice" %in% names)
  expect_equal(model$metrics[[which(names == "surface_dice")]]$tau, 1.5)
})

test_that("the generalized Dice loss is admissible wherever a class loss is", {
  skip_if_no_engine()
  expect_true("generalized_dice" %in% available_losses()$name)
  expect_true("generalized_dice" %in% available_losses("instance_segmentation")$name)

  # Its own `loss` at the instance task - the mask is classes at either separation - and as
  # the region term of `loss_boundary()`, which is the combination `?loss_boundary` now
  # recommends and therefore the one worth proving reaches the engine.
  instance <- built(list(
    task = "instance_segmentation",
    data = list(train_path = ".", validation_path = "."),
    models = list(list(name = "m", input_shape = c(64L, 64L),
                       separation = list(name = "rim", width = 2L),
                       loss = loss_generalized_dice()))
  ))
  expect_true(isTRUE(instance$ok), info = paste(unlist(instance$message), collapse = " "))

  wrapped <- built(semantic_config(model = list(
    loss = loss_boundary(region = loss_generalized_dice(), alpha = 0.9)
  )))
  expect_true(isTRUE(wrapped$ok), info = paste(unlist(wrapped$message), collapse = " "))
  region <- rplatypus:::shim()$spec_as_dict(wrapped$spec)$models[[1]]$loss$region
  expect_identical(region$name, "generalized_dice")
})

test_that("a flow field refuses it by name, because it scores classes", {
  # The other half of the applicability question, and the half a listing cannot answer: a
  # loss can be in every listing and still be refused everywhere somebody would reach for
  # it. `separation_flows(loss = )` is the one place the restriction runs the other way.
  skip_if_no_engine()
  refused <- built(list(
    task = "instance_segmentation",
    data = list(train_path = ".", validation_path = "."),
    models = list(list(name = "m", input_shape = c(64L, 64L),
                       separation = list(name = "flows",
                                         loss = loss_generalized_dice())))
  ))
  expect_false(isTRUE(refused$ok))
  expect_match(paste(unlist(refused$message), collapse = " "), "generalized_dice")

  # The control, because a refusal is only about this loss if the same shape is accepted
  # with a continuous one in its place: §4's probe that answers for its own reasons.
  accepted <- built(list(
    task = "instance_segmentation",
    data = list(train_path = ".", validation_path = "."),
    models = list(list(name = "m", input_shape = c(64L, 64L),
                       separation = list(name = "flows", loss = loss_mse())))
  ))
  expect_true(isTRUE(accepted$ok), info = paste(unlist(accepted$message), collapse = " "))
})

test_that("`?loss_boundary` names the region losses the engine's own union holds", {
  # This page said "the nine region losses" until the pin moved and there were ten. A count
  # in prose with nothing holding it is §3C's forbidden number, so the sentence states the
  # rule instead - every loss except the three continuous ones and `boundary` itself - and
  # this is what holds the rule. The authority is the discriminator the engine publishes for
  # `BoundaryLoss.region`, which is a different object from `available_losses()`.
  skip_if_no_engine()
  region <- rplatypus:::engine()$spec_schema()$`$defs`$BoundaryLoss$properties$region
  theirs <- sort(names(region$discriminator$mapping))
  expect_gt(length(theirs), 5)

  continuous <- c("mse", "l1", "smooth_l1")
  expected <- sort(setdiff(available_losses("instance_segmentation")$name,
                           c("boundary", continuous)))
  expect_identical(theirs, expected)

  # And R exports a constructor for every one of them, which is what makes the sentence a
  # statement about this package rather than about the engine.
  expect_identical(
    setdiff(paste0("loss_", theirs), getNamespaceExports("rplatypus")),
    character(0)
  )
})

test_that("`?losses` reports the generalized Dice table the engine measured", {
  # §3C: a measured number in prose needs a test holding it to the measurement. R cannot
  # re-measure this one - there is no way to evaluate a loss from this side - so it is held
  # to the engine's own text, which the engine's suite holds to the measurement. Two links,
  # each with a test under it, rather than a number nothing reads.
  skip_if_no_engine()
  page <- rd_text("losses")
  expect_false(is.na(page))

  reticulate::py_run_string(
    "from pyplatypus.spec.components import GeneralizedDiceLoss as _g\ntheirs = _g.__doc__"
  )
  theirs <- reticulate::py_eval("theirs")
  expect_true(nzchar(theirs %||% ""),
              info = "no docstring reached: the pin or the probe is wrong")

  figures <- c("0.1658", "0.4731", "0.1600", "0.1629", "0.3166", "0.3278", "0.9511", "9801")
  mine <- figures[!vapply(figures, grepl, logical(1), x = page, fixed = TRUE)]
  expect_identical(mine, character(0),
                   info = "?losses stopped printing a figure this test is about")
  absent <- figures[!vapply(figures, grepl, logical(1), x = theirs, fixed = TRUE)]
  expect_identical(
    absent, character(0),
    info = paste("?losses prints figures the engine no longer reports:",
                 paste(absent, collapse = ", "), "- re-read the engine's docstring")
  )
})

test_that("`?metrics` reports the surface Dice background numbers the engine measured", {
  # The same arrangement one layer down, and against the schema rather than a docstring:
  # these four numbers are in the engine's `description` for `include_background`, which is
  # published documentation on its side and is what the field-prose ledger digests. So this
  # is the ledger's question asked about the numbers rather than about the text.
  skip_if_no_engine()
  page <- rd_text("metrics")
  expect_false(is.na(page))

  theirs <- rplatypus:::engine()$spec_schema()$`$defs`$SurfaceDiceMetric$
    properties$include_background$description
  expect_true(nzchar(theirs %||% ""))

  for (figure in c("0.6110", "0.7030", "0.5583", "0.6266")) {
    expect_true(grepl(figure, page, fixed = TRUE), info = paste("?metrics:", figure))
    expect_true(grepl(figure, theirs, fixed = TRUE), info = paste("the engine:", figure))
  }
})

test_that("a tolerance that cannot mean anything is refused here, before a run", {
  # R's own refusal, and it is not duplication: the engine refuses `tau` by its schema and
  # that refusal arrives when a specification is built. Somebody assembling a metric list in
  # a script gets it at the line they wrote instead.
  expect_error(metric_surface_dice(), "tau")
  expect_error(metric_surface_dice(tau = 0), "one positive number")
  expect_error(metric_surface_dice(tau = -1), "one positive number")
  expect_error(metric_surface_dice(tau = NA_real_), "one positive number")
  expect_error(metric_surface_dice(tau = c(1, 2)), "one positive number")
  expect_error(metric_surface_dice(tau = "1.5"), "one positive number")
  expect_identical(metric_surface_dice(tau = 1.5)$tau, 1.5)
})

test_that("the engine refuses the same tolerances, so R's message is a convenience", {
  # The control for the test above: if the engine accepted `tau = 0`, R's refusal would be
  # this package inventing a rule, which §3D.4 says it must not. Asked by handing the engine
  # the field directly, since R's constructor cannot build the bad case.
  skip_if_no_engine()
  for (tau in list(0, -1)) {
    result <- built(semantic_config(model = list(
      metrics = list(list(name = "surface_dice", tau = tau))
    )))
    expect_false(isTRUE(result$ok), info = paste("tau =", tau))
    expect_match(paste(unlist(result$message), collapse = " "), "tau")
  }
  # And an absent tolerance is refused rather than defaulted, which is the claim `?metrics`
  # makes about why there is no default on this side either.
  absent <- built(semantic_config(model = list(
    metrics = list(list(name = "surface_dice"))
  )))
  expect_false(isTRUE(absent$ok))
  expect_match(paste(unlist(absent$message), collapse = " "), "tau")
})
