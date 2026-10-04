# Losses, metrics, optimisers and callbacks.
#
# Named the way the old platypus and the R keras package name things - `loss_dice()`,
# `metric_iou()`, `optimizer_adam()` - because R has one namespace per package and a
# `dice()` that is a loss cannot coexist with a `dice()` that is a metric. The prefixes
# also make the whole family discoverable by typing `loss_` and waiting.
#
# Integers are coerced here rather than left to pydantic. R numbers are doubles, so a
# plain `4` arrives in Python as `4.0`; relying on coercion works until someone writes
# `blocks = 4.5` and gets a puzzling message from a library they never invoked.

int1 <- function(x) if (is.null(x)) NULL else as.integer(x)

#' Loss functions
#'
#' The objective the model is trained against. Every one of these works unchanged on
#' volumes as well as images.
#'
#' @param smooth Added to numerator and denominator to keep an empty class finite.
#' @param gamma For focal losses, how hard to discount the pixels already classified well.
#' @param alpha For Tversky, the weight on false negatives; raising it buys recall.
#'   Note that `alpha = 0.5` is Dice exactly only when `smooth = 0`.
#' @param label_smoothing,cce_weight,ce_ratio,per_image See the individual losses.
#' @return A loss specification, to be passed to a model constructor.
#' @name losses
#' @examples
#' loss_focal(gamma = 2)
#' loss_tversky(alpha = 0.7)
NULL

#' @rdname losses
#' @export
loss_iou <- function(smooth = 1) list(name = "iou", smooth = smooth)

#' @rdname losses
#' @export
loss_dice <- function(smooth = 1) list(name = "dice", smooth = smooth)

#' @rdname losses
#' @export
loss_cce <- function(label_smoothing = 0) {
  list(name = "cce", label_smoothing = label_smoothing)
}

#' @rdname losses
#' @export
loss_cce_dice <- function(cce_weight = 0.5, smooth = 1) {
  list(name = "cce_dice", cce_weight = cce_weight, smooth = smooth)
}

#' @rdname losses
#' @export
loss_focal <- function(gamma = 2, alpha = NULL) {
  list(name = "focal", gamma = gamma, alpha = alpha)
}

#' @rdname losses
#' @export
loss_tversky <- function(alpha = 0.5, smooth = 1) {
  list(name = "tversky", alpha = alpha, smooth = smooth)
}

#' @rdname losses
#' @export
loss_focal_tversky <- function(alpha = 0.5, gamma = 1, smooth = 1) {
  list(name = "focal_tversky", alpha = alpha, gamma = gamma, smooth = smooth)
}

#' @rdname losses
#' @export
loss_combo <- function(alpha = 0.5, ce_ratio = 0.5) {
  list(name = "combo", alpha = alpha, ce_ratio = ce_ratio)
}

#' @rdname losses
#' @export
loss_lovasz <- function(per_image = FALSE) list(name = "lovasz", per_image = per_image)

#' Metrics
#'
#' What gets reported. Unlike the losses these are computed on the hard prediction - the
#' mask you will actually be handed - because a metric taken on probabilities reads higher
#' than the model deserves.
#'
#' @param smooth Smoothing. Zero is allowed and is the honest choice for a reported
#'   number: with smoothing, a class absent from an image scores a perfect 1.
#' @param include_background Average over the background class as well. In medical images
#'   the background is usually most of the picture, so leaving it in can turn a model that
#'   found nothing into one that looks respectable. Papers report foreground only.
#' @param alpha For Tversky, the weight on false negatives.
#' @return A metric specification.
#' @name metrics
#' @examples
#' metric_dice(include_background = FALSE)
NULL

#' @rdname metrics
#' @export
metric_iou <- function(smooth = 1, include_background = TRUE) {
  list(name = "iou", smooth = smooth, include_background = include_background)
}

#' @rdname metrics
#' @export
metric_dice <- function(smooth = 1, include_background = TRUE) {
  list(name = "dice", smooth = smooth, include_background = include_background)
}

#' @rdname metrics
#' @export
metric_tversky <- function(alpha = 0.5, smooth = 1, include_background = TRUE) {
  list(name = "tversky", alpha = alpha, smooth = smooth,
       include_background = include_background)
}

#' Optimisers
#'
#' @param learning_rate,weight_decay Standard.
#' @param beta_1,beta_2,eps,amsgrad,momentum,nesterov,alpha,rho,lr_decay Per optimiser.
#' @return An optimiser specification.
#' @name optimizers
#' @examples
#' optimizer_adam(learning_rate = 1e-4)
NULL

#' @rdname optimizers
#' @export
optimizer_adam <- function(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999,
                           eps = 1e-8, weight_decay = 0, amsgrad = FALSE) {
  list(name = "adam", learning_rate = learning_rate, beta_1 = beta_1, beta_2 = beta_2,
       eps = eps, weight_decay = weight_decay, amsgrad = amsgrad)
}

#' @rdname optimizers
#' @export
optimizer_adamw <- function(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999,
                            eps = 1e-8, weight_decay = 1e-2) {
  list(name = "adamw", learning_rate = learning_rate, beta_1 = beta_1, beta_2 = beta_2,
       eps = eps, weight_decay = weight_decay)
}

#' @rdname optimizers
#' @export
optimizer_sgd <- function(learning_rate = 1e-3, momentum = 0, nesterov = FALSE,
                          weight_decay = 0) {
  list(name = "sgd", learning_rate = learning_rate, momentum = momentum,
       nesterov = nesterov, weight_decay = weight_decay)
}

#' @rdname optimizers
#' @export
optimizer_rmsprop <- function(learning_rate = 1e-3, alpha = 0.99, momentum = 0,
                              eps = 1e-8, weight_decay = 0) {
  list(name = "rmsprop", learning_rate = learning_rate, alpha = alpha,
       momentum = momentum, eps = eps, weight_decay = weight_decay)
}

#' @rdname optimizers
#' @export
optimizer_adagrad <- function(learning_rate = 1e-3, lr_decay = 0, eps = 1e-10,
                              weight_decay = 0) {
  list(name = "adagrad", learning_rate = learning_rate, lr_decay = lr_decay,
       eps = eps, weight_decay = weight_decay)
}

#' @rdname optimizers
#' @export
optimizer_adadelta <- function(learning_rate = 1e-3, rho = 0.9, eps = 1e-6,
                               weight_decay = 0) {
  list(name = "adadelta", learning_rate = learning_rate, rho = rho, eps = eps,
       weight_decay = weight_decay)
}

#' @rdname optimizers
#' @export
optimizer_adamax <- function(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999,
                             eps = 1e-8, weight_decay = 0) {
  list(name = "adamax", learning_rate = learning_rate, beta_1 = beta_1,
       beta_2 = beta_2, eps = eps, weight_decay = weight_decay)
}

#' @rdname optimizers
#' @export
optimizer_nadam <- function(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999,
                            eps = 1e-8, weight_decay = 0) {
  list(name = "nadam", learning_rate = learning_rate, beta_1 = beta_1,
       beta_2 = beta_2, eps = eps, weight_decay = weight_decay)
}

#' Callbacks
#'
#' Things that happen between epochs. `monitor` is checked against the metrics the model
#' actually asks for, so a callback watching a number that will never arrive is refused
#' when the specification is built rather than waited on forever.
#'
#' Whether the watched quantity should rise or fall is worked out from its name: anything
#' ending in `loss` is minimised, everything else maximised.
#'
#' @section Cosine annealing:
#'
#' `callback_cosine_annealing()` decays the learning rate from its initial value to `min_lr`
#' along a cosine. It watches nothing: it is a function of how far through the run you are
#' rather than of how the run is going, which is the difference from
#' `callback_reduce_lr_on_plateau()` - and a run may legitimately want both.
#'
#' Each parameter group decays from its **own** initial rate, so an `encoder_learning_rate`
#' set on a model is not flattened at the first epoch. That would have been invisible: the
#' history records one `learning_rate`, the first group's.
#'
#' `epochs` unset means the model's own `epochs`, which is what you want - a cosine that
#' ends where the run ends.
#'
#' @param monitor What to watch: `"val_loss"`, `"train_loss"`, or `"val_"` followed by a
#'   metric name, such as `"val_dice"`.
#' @param patience Epochs to wait before acting.
#' @param min_delta Improvement smaller than this does not count.
#' @param restore_best Put the best weights back when training stops.
#' @param path Where to write.
#' @param save_best_only Only write when the watched quantity improves.
#' @param factor,min_lr For the learning rate schedule. `min_lr` is the floor for both
#'   `callback_reduce_lr_on_plateau()` and `callback_cosine_annealing()`.
#' @param epochs For `callback_cosine_annealing()`, how many epochs to spread the decay
#'   over. Unset means the model's `epochs`.
#' @return A callback specification.
#' @name callbacks
#' @examples
#' callback_early_stopping(monitor = "val_dice", patience = 10)
NULL

#' @rdname callbacks
#' @export
callback_early_stopping <- function(monitor = "val_loss", patience = 10,
                                    min_delta = 0, restore_best = TRUE) {
  list(name = "early_stopping", monitor = monitor, patience = int1(patience),
       min_delta = min_delta, restore_best = restore_best)
}

#' @rdname callbacks
#' @export
callback_model_checkpoint <- function(path, monitor = "val_loss",
                                      save_best_only = TRUE) {
  list(name = "model_checkpoint", path = path, monitor = monitor,
       save_best_only = save_best_only)
}

#' @rdname callbacks
#' @export
callback_reduce_lr_on_plateau <- function(monitor = "val_loss", factor = 0.1,
                                          patience = 5, min_lr = 0) {
  list(name = "reduce_lr_on_plateau", monitor = monitor, factor = factor,
       patience = int1(patience), min_lr = min_lr)
}

#' @rdname callbacks
#' @export
callback_cosine_annealing <- function(min_lr = 0, epochs = NULL) {
  compact(list(name = "cosine_annealing", min_lr = min_lr,
               epochs = if (is.null(epochs)) NULL else int1(epochs)))
}

#' @rdname callbacks
#' @export
callback_csv_logger <- function(path) list(name = "csv_logger", path = path)

#' @rdname callbacks
#' @export
callback_terminate_on_nan <- function() list(name = "terminate_on_nan")

#' An augmentation step
#'
#' Names come from the installed 'albumentations', so whatever it offers is available and
#' a misspelling is caught when the specification is built, with a suggestion.
#'
#' @param name The transform, for instance `"HorizontalFlip"`.
#' @param ... Its parameters, for instance `p = 0.5`.
#' @return An augmentation specification.
#' @export
#' @examples
#' augment("HorizontalFlip", p = 0.5)
augment <- function(name, ...) {
  list(name = name, params = list(...))
}


#' Which augmentations are available
#'
#' The transforms the installed 'albumentations' offers, and - for volumes - the ones it can
#' actually apply to them.
#'
#' Worth asking rather than guessing. Support for volumes is uneven: most geometric transforms
#' work and several intensity ones raise from inside the library, so `rank = 3` returns a
#' shorter list than `rank = 2`. A transform outside it is refused when the specification is
#' built, by name, rather than failing an hour into training.
#'
#' @param rank 2 for images, 3 for volumes.
#' @param pattern Optional regular expression to filter the names, for browsing: `"Flip"`,
#'   `"3D$"`, `"Elastic|Grid"`.
#' @return A character vector of transform names.
#' @seealso [augment()], which uses one.
#' @examples
#' \dontrun{
#' available_augmentations()                       # everything, for images
#' available_augmentations(rank = 3)               # what volumes can take
#' available_augmentations(rank = 3, pattern = "Flip|Crop")
#' setdiff(available_augmentations(), available_augmentations(rank = 3))   # the gap
#' }
#' @export
available_augmentations <- function(rank = 2, pattern = NULL) {
  if (!rank %in% c(2, 3)) {
    stop("`rank` is 2 for images or 3 for volumes.", call. = FALSE)
  }
  result <- shim()$transform_names(rank = as.integer(rank))
  if (!isTRUE(result$ok)) abort_engine(result)

  names <- unlist(result$transforms)
  if (!is.null(pattern)) names <- grep(pattern, names, value = TRUE)
  names
}
