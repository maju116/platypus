# Model constructors.
#
# Four architectures from one builder, because they differ in three decisions - whether
# the convolution block is residual, whether skips are concatenated or added, and whether
# the skip paths are nested - and every other switch composes with all of them.
#
# The old package expressed this as three independent flags, so it also built the
# combinations nobody named and nobody tested. Here you ask for one architecture by name.

#' Segmentation models
#'
#' Four members of the U-shaped family. They take the same arguments and differ only in
#' what happens between the contracting and expanding paths: `u_net` concatenates its skip
#' connections, `linknet` adds them, `res_u_net` makes each block residual, and
#' `u_net_plus_plus` fills in the nested skip pathways.
#'
#' `input_shape` is the tile the model sees: two numbers for images, three for volumes.
#' Its length is what makes a model 2D or 3D - there is no separate switch. Every spatial
#' dimension must divide by `2^blocks`, or the decoder cannot line up with the encoder,
#' and that is checked when the specification is built rather than part-way through
#' training.
#'
#' @param name Unique within a specification; names the outputs and the table rows.
#' @param input_shape Tile size: `c(height, width)`, or `c(depth, height, width)`.
#' @param channels Input channels. 3 for colour, 1 for greyscale.
#' @param n_class Number of classes, including background. Must match the colormap.
#' @param blocks How many times the resolution is halved.
#' @param filters Filters in the first block; doubled at each level.
#' @param block_width Convolutions per block.
#' @param dropout Dropout rate, 0 to switch it off.
#' @param batch_normalization Normalise between convolutions.
#' @param separable_conv Depthwise-separable convolutions. Far fewer parameters, a little
#'   slower to converge.
#' @param spatial_dropout Drop whole feature maps rather than individual activations.
#' @param upsample Interpolate and convolve instead of using a transposed convolution.
#' @param deep_supervision Train every decoder depth, not only the last. `u_net_plus_plus`
#'   only, since it is the one with intermediate outputs to supervise.
#' @param activation One of `"relu"`, `"leaky_relu"`, `"elu"`, `"selu"`, `"gelu"`,
#'   `"silu"`, `"tanh"`.
#' @param initialiser One of `"he_normal"`, `"he_uniform"`, `"glorot_normal"`,
#'   `"glorot_uniform"`.
#' @param loss A loss, see [losses].
#' @param metrics A list of metrics, see [metrics].
#' @param optimizer An optimiser, see [optimizers].
#' @param callbacks A list of callbacks, see [callbacks].
#' @param augmentation A list of [augment()] steps, applied to training data only.
#' @param epochs,batch_size Training length and batch size.
#' @param splits Cut each image into a grid of tiles instead of shrinking it: one number
#'   per spatial dimension, so `c(2, 3)` gives six tiles. Use it to segment a large image
#'   without throwing away resolution; predictions are reassembled to the original size.
#' @param weights A checkpoint to start from, or to use as-is with `fit = FALSE`.
#' @param fit Set `FALSE` to load `weights` and skip training.
#' @param encoder A pretrained-architecture backbone to use as the contracting path
#'   instead of the built-in encoder, named as timm names it: `"resnet34"`,
#'   `"efficientnet_b0"`, `"mobilenetv3_large_100"`, `"densenet121"`, `"vgg16"` and the
#'   other convolutional families. Naming one builds that architecture; it does not load
#'   anything. **2D only** - ImageNet is images, so a backbone with a 3D `input_shape` is
#'   refused while the specification is read. The patch-based families (`convnext_*`,
#'   `swin_*`) are refused too: their features start at a quarter of the input, and the
#'   half-resolution level a U-shaped decoder recovers fine detail from does not exist.
#'   `blocks` then says how many of the backbone's stages to use.
#' @param pretrained Load the backbone's ImageNet weights. Separate from `encoder` on
#'   purpose, so naming an architecture never reaches the network on a machine that has
#'   none. Measured on three datasets, transfer does not beat the built-in encoder here -
#'   see [encoders] for the table and what it is actually good for.
#' @param freeze_encoder Hold the transferred layers still for this many epochs, then
#'   train them. **Use this rather than `encoder_learning_rate` alone**: it was the better
#'   of the two on all three datasets measured. A number at or above `epochs` keeps them
#'   fixed for the whole run.
#' @param encoder_learning_rate A separate, smaller rate for the layers that arrived
#'   pretrained. The full-resolution stage in front of them is platypus's own and starts
#'   random, so it keeps the optimiser's rate. Giving this instead of `freeze_encoder`
#'   measured worse everywhere, and on one dataset worse than no backbone at all.
#' @return A model specification, to be passed to [platypus_spec()].
#' @name models
#' @examples
#' u_net("unet", input_shape = c(256, 256), blocks = 4, filters = 16)
#'
#' # A large image, tiled rather than shrunk
#' u_net("hd", input_shape = c(256, 256), splits = c(4, 6))
NULL

# One builder, four names. The signatures are spelled out in the wrappers rather than
# funnelled through `...` so that typing `u_net(` in an editor offers every knob - the
# difference between a package someone explores and one they have to read about first.
segmentation_model <- function(architecture, name, input_shape, channels, n_class,
                               blocks, filters, block_width, dropout,
                               batch_normalization, separable_conv, spatial_dropout,
                               upsample, deep_supervision, activation, initialiser,
                               loss, metrics, optimizer, callbacks, augmentation,
                               epochs, batch_size, splits, weights, fit,
                               encoder, pretrained, freeze_encoder,
                               encoder_learning_rate) {
  list(
    name = name,
    architecture = architecture,
    input_shape = as.integer(input_shape),
    channels = int1(channels),
    n_class = int1(n_class),
    blocks = int1(blocks),
    filters = int1(filters),
    block_width = int1(block_width),
    dropout = dropout,
    batch_normalization = batch_normalization,
    separable_conv = separable_conv,
    spatial_dropout = spatial_dropout,
    upsample = upsample,
    deep_supervision = deep_supervision,
    activation = activation,
    initialiser = initialiser,
    loss = loss,
    metrics = metrics,
    optimizer = optimizer,
    callbacks = callbacks,
    augmentation = augmentation,
    epochs = int1(epochs),
    batch_size = int1(batch_size),
    splits = if (is.null(splits)) NULL else as.integer(splits),
    weights = weights,
    fit = fit,
    # All four stay NULL unless asked for, and `compact()` drops a NULL before the
    # request is built. An engine that has never heard of these keeps working for
    # everyone who is not asking, and only the person who asks meets the requirement.
    encoder = encoder,
    pretrained = pretrained,
    freeze_encoder = if (is.null(freeze_encoder)) NULL else int1(freeze_encoder),
    encoder_learning_rate = encoder_learning_rate
  )
}

#' @rdname models
#' @export
u_net <- function(name, input_shape, channels = 3, n_class = 2, blocks = 4,
                   filters = 16, block_width = 2, dropout = 0,
                   batch_normalization = TRUE, separable_conv = FALSE,
                   spatial_dropout = TRUE, upsample = FALSE,
                   deep_supervision = FALSE, activation = "relu",
                   initialiser = "he_normal", loss = loss_cce(),
                   metrics = list(metric_iou()), optimizer = optimizer_adam(),
                   callbacks = list(), augmentation = NULL, epochs = 10,
                   batch_size = 8, splits = NULL, weights = NULL, fit = TRUE,
                   encoder = NULL, pretrained = NULL, freeze_encoder = NULL,
                   encoder_learning_rate = NULL) {
  segmentation_model(
    architecture = "u_net",
    name = name,
    input_shape = input_shape,
    channels = channels,
    n_class = n_class,
    blocks = blocks,
    filters = filters,
    block_width = block_width,
    dropout = dropout,
    batch_normalization = batch_normalization,
    separable_conv = separable_conv,
    spatial_dropout = spatial_dropout,
    upsample = upsample,
    deep_supervision = deep_supervision,
    activation = activation,
    initialiser = initialiser,
    loss = loss,
    metrics = metrics,
    optimizer = optimizer,
    callbacks = callbacks,
    augmentation = augmentation,
    epochs = epochs,
    batch_size = batch_size,
    splits = splits,
    weights = weights,
    fit = fit,
    encoder = encoder,
    pretrained = pretrained,
    freeze_encoder = freeze_encoder,
    encoder_learning_rate = encoder_learning_rate
  )
}

#' @rdname models
#' @export
u_net_plus_plus <- function(name, input_shape, channels = 3, n_class = 2, blocks = 4,
                             filters = 16, block_width = 2, dropout = 0,
                             batch_normalization = TRUE, separable_conv = FALSE,
                             spatial_dropout = TRUE, upsample = FALSE,
                             deep_supervision = FALSE, activation = "relu",
                             initialiser = "he_normal", loss = loss_cce(),
                             metrics = list(metric_iou()), optimizer = optimizer_adam(),
                             callbacks = list(), augmentation = NULL, epochs = 10,
                             batch_size = 8, splits = NULL, weights = NULL, fit = TRUE,
                   encoder = NULL, pretrained = NULL, freeze_encoder = NULL,
                   encoder_learning_rate = NULL) {
  segmentation_model(
    architecture = "u_net_plus_plus",
    name = name,
    input_shape = input_shape,
    channels = channels,
    n_class = n_class,
    blocks = blocks,
    filters = filters,
    block_width = block_width,
    dropout = dropout,
    batch_normalization = batch_normalization,
    separable_conv = separable_conv,
    spatial_dropout = spatial_dropout,
    upsample = upsample,
    deep_supervision = deep_supervision,
    activation = activation,
    initialiser = initialiser,
    loss = loss,
    metrics = metrics,
    optimizer = optimizer,
    callbacks = callbacks,
    augmentation = augmentation,
    epochs = epochs,
    batch_size = batch_size,
    splits = splits,
    weights = weights,
    fit = fit,
    encoder = encoder,
    pretrained = pretrained,
    freeze_encoder = freeze_encoder,
    encoder_learning_rate = encoder_learning_rate
  )
}

#' @rdname models
#' @export
res_u_net <- function(name, input_shape, channels = 3, n_class = 2, blocks = 4,
                       filters = 16, block_width = 2, dropout = 0,
                       batch_normalization = TRUE, separable_conv = FALSE,
                       spatial_dropout = TRUE, upsample = FALSE,
                       deep_supervision = FALSE, activation = "relu",
                       initialiser = "he_normal", loss = loss_cce(),
                       metrics = list(metric_iou()), optimizer = optimizer_adam(),
                       callbacks = list(), augmentation = NULL, epochs = 10,
                       batch_size = 8, splits = NULL, weights = NULL, fit = TRUE,
                   encoder = NULL, pretrained = NULL, freeze_encoder = NULL,
                   encoder_learning_rate = NULL) {
  segmentation_model(
    architecture = "res_u_net",
    name = name,
    input_shape = input_shape,
    channels = channels,
    n_class = n_class,
    blocks = blocks,
    filters = filters,
    block_width = block_width,
    dropout = dropout,
    batch_normalization = batch_normalization,
    separable_conv = separable_conv,
    spatial_dropout = spatial_dropout,
    upsample = upsample,
    deep_supervision = deep_supervision,
    activation = activation,
    initialiser = initialiser,
    loss = loss,
    metrics = metrics,
    optimizer = optimizer,
    callbacks = callbacks,
    augmentation = augmentation,
    epochs = epochs,
    batch_size = batch_size,
    splits = splits,
    weights = weights,
    fit = fit,
    encoder = encoder,
    pretrained = pretrained,
    freeze_encoder = freeze_encoder,
    encoder_learning_rate = encoder_learning_rate
  )
}

#' @rdname models
#' @export
linknet <- function(name, input_shape, channels = 3, n_class = 2, blocks = 4,
                     filters = 16, block_width = 2, dropout = 0,
                     batch_normalization = TRUE, separable_conv = FALSE,
                     spatial_dropout = TRUE, upsample = FALSE,
                     deep_supervision = FALSE, activation = "relu",
                     initialiser = "he_normal", loss = loss_cce(),
                     metrics = list(metric_iou()), optimizer = optimizer_adam(),
                     callbacks = list(), augmentation = NULL, epochs = 10,
                     batch_size = 8, splits = NULL, weights = NULL, fit = TRUE,
                   encoder = NULL, pretrained = NULL, freeze_encoder = NULL,
                   encoder_learning_rate = NULL) {
  segmentation_model(
    architecture = "linknet",
    name = name,
    input_shape = input_shape,
    channels = channels,
    n_class = n_class,
    blocks = blocks,
    filters = filters,
    block_width = block_width,
    dropout = dropout,
    batch_normalization = batch_normalization,
    separable_conv = separable_conv,
    spatial_dropout = spatial_dropout,
    upsample = upsample,
    deep_supervision = deep_supervision,
    activation = activation,
    initialiser = initialiser,
    loss = loss,
    metrics = metrics,
    optimizer = optimizer,
    callbacks = callbacks,
    augmentation = augmentation,
    epochs = epochs,
    batch_size = batch_size,
    splits = splits,
    weights = weights,
    fit = fit,
    encoder = encoder,
    pretrained = pretrained,
    freeze_encoder = freeze_encoder,
    encoder_learning_rate = encoder_learning_rate
  )
}
