# Segmentation models

Four members of the U-shaped family. They take the same arguments and
differ only in what happens between the contracting and expanding paths:
`u_net` concatenates its skip connections, `linknet` adds them,
`res_u_net` makes each block residual, and `u_net_plus_plus` fills in
the nested skip pathways.

## Usage

``` r
u_net(
  name,
  input_shape,
  channels = 3,
  n_class = 2,
  blocks = 4,
  filters = 16,
  block_width = 2,
  dropout = 0,
  batch_normalization = TRUE,
  separable_conv = FALSE,
  spatial_dropout = TRUE,
  upsample = FALSE,
  deep_supervision = FALSE,
  activation = "relu",
  initialiser = "he_normal",
  loss = loss_cce(),
  metrics = list(metric_iou()),
  optimizer = optimizer_adam(),
  callbacks = list(),
  augmentation = NULL,
  epochs = 10,
  batch_size = 8,
  splits = NULL,
  weights = NULL,
  fit = TRUE,
  encoder = NULL,
  pretrained = NULL,
  freeze_encoder = NULL,
  encoder_learning_rate = NULL
)

u_net_plus_plus(
  name,
  input_shape,
  channels = 3,
  n_class = 2,
  blocks = 4,
  filters = 16,
  block_width = 2,
  dropout = 0,
  batch_normalization = TRUE,
  separable_conv = FALSE,
  spatial_dropout = TRUE,
  upsample = FALSE,
  deep_supervision = FALSE,
  activation = "relu",
  initialiser = "he_normal",
  loss = loss_cce(),
  metrics = list(metric_iou()),
  optimizer = optimizer_adam(),
  callbacks = list(),
  augmentation = NULL,
  epochs = 10,
  batch_size = 8,
  splits = NULL,
  weights = NULL,
  fit = TRUE,
  encoder = NULL,
  pretrained = NULL,
  freeze_encoder = NULL,
  encoder_learning_rate = NULL
)

res_u_net(
  name,
  input_shape,
  channels = 3,
  n_class = 2,
  blocks = 4,
  filters = 16,
  block_width = 2,
  dropout = 0,
  batch_normalization = TRUE,
  separable_conv = FALSE,
  spatial_dropout = TRUE,
  upsample = FALSE,
  deep_supervision = FALSE,
  activation = "relu",
  initialiser = "he_normal",
  loss = loss_cce(),
  metrics = list(metric_iou()),
  optimizer = optimizer_adam(),
  callbacks = list(),
  augmentation = NULL,
  epochs = 10,
  batch_size = 8,
  splits = NULL,
  weights = NULL,
  fit = TRUE,
  encoder = NULL,
  pretrained = NULL,
  freeze_encoder = NULL,
  encoder_learning_rate = NULL
)

linknet(
  name,
  input_shape,
  channels = 3,
  n_class = 2,
  blocks = 4,
  filters = 16,
  block_width = 2,
  dropout = 0,
  batch_normalization = TRUE,
  separable_conv = FALSE,
  spatial_dropout = TRUE,
  upsample = FALSE,
  deep_supervision = FALSE,
  activation = "relu",
  initialiser = "he_normal",
  loss = loss_cce(),
  metrics = list(metric_iou()),
  optimizer = optimizer_adam(),
  callbacks = list(),
  augmentation = NULL,
  epochs = 10,
  batch_size = 8,
  splits = NULL,
  weights = NULL,
  fit = TRUE,
  encoder = NULL,
  pretrained = NULL,
  freeze_encoder = NULL,
  encoder_learning_rate = NULL
)
```

## Arguments

- name:

  Unique within a specification; names the outputs and the table rows.

- input_shape:

  Tile size: `c(height, width)`, or `c(depth, height, width)`.

- channels:

  Input channels. 3 for colour, 1 for greyscale.

- n_class:

  Number of classes, including background. Must match the colormap.

- blocks:

  How many times the resolution is halved.

- filters:

  Filters in the first block; doubled at each level.

- block_width:

  Convolutions per block.

- dropout:

  Dropout rate, 0 to switch it off.

- batch_normalization:

  Normalise between convolutions.

- separable_conv:

  Depthwise-separable convolutions. Far fewer parameters, a little
  slower to converge.

- spatial_dropout:

  Drop whole feature maps rather than individual activations.

- upsample:

  Interpolate and convolve instead of using a transposed convolution.

- deep_supervision:

  Train every decoder depth, not only the last. `u_net_plus_plus` only,
  since it is the one with intermediate outputs to supervise.

- activation:

  One of `"relu"`, `"leaky_relu"`, `"elu"`, `"selu"`, `"gelu"`,
  `"silu"`, `"tanh"`.

- initialiser:

  One of `"he_normal"`, `"he_uniform"`, `"glorot_normal"`,
  `"glorot_uniform"`.

- loss:

  A loss, see
  [losses](https://maju116.github.io/platypus/reference/losses.md).

- metrics:

  A list of metrics, see
  [metrics](https://maju116.github.io/platypus/reference/metrics.md).

- optimizer:

  An optimiser, see
  [optimizers](https://maju116.github.io/platypus/reference/optimizers.md).

- callbacks:

  A list of callbacks, see
  [callbacks](https://maju116.github.io/platypus/reference/callbacks.md).

- augmentation:

  A list of
  [`augment()`](https://maju116.github.io/platypus/reference/augment.md)
  steps, applied to training data only.

- epochs, batch_size:

  Training length and batch size.

- splits:

  Cut each image into a grid of tiles instead of shrinking it: one
  number per spatial dimension, so `c(2, 3)` gives six tiles. Use it to
  segment a large image without throwing away resolution; predictions
  are reassembled to the original size.

- weights:

  A checkpoint to start from, or to use as-is with `fit = FALSE`.

- fit:

  Set `FALSE` to load `weights` and skip training.

- encoder:

  A pretrained-architecture backbone to use as the contracting path
  instead of the built-in encoder, named as timm names it: `"resnet34"`,
  `"efficientnet_b0"`, `"mobilenetv3_large_100"`, `"densenet121"`,
  `"vgg16"` and the other convolutional families. Naming one builds that
  architecture; it does not load anything. **2D only** - ImageNet is
  images, so a backbone with a 3D `input_shape` is refused while the
  specification is read. The patch-based families (`convnext_*`,
  `swin_*`) are refused too: their features start at a quarter of the
  input, and the half-resolution level a U-shaped decoder recovers fine
  detail from does not exist. `blocks` then says how many of the
  backbone's stages to use.

- pretrained:

  Load the backbone's ImageNet weights. Separate from `encoder` on
  purpose, so naming an architecture never reaches the network on a
  machine that has none. Measured on three datasets, transfer does not
  beat the built-in encoder here - see
  [encoders](https://maju116.github.io/platypus/reference/encoders.md)
  for the table and what it is actually good for.

- freeze_encoder:

  Hold the transferred layers still for this many epochs, then train
  them. **Use this rather than `encoder_learning_rate` alone**: it was
  the better of the two on all three datasets measured. A number at or
  above `epochs` keeps them fixed for the whole run.

- encoder_learning_rate:

  A separate, smaller rate for the layers that arrived pretrained. The
  full-resolution stage in front of them is platypus's own and starts
  random, so it keeps the optimiser's rate. Giving this instead of
  `freeze_encoder` measured worse everywhere, and on one dataset worse
  than no backbone at all.

## Value

A model specification, to be passed to
[`platypus_spec()`](https://maju116.github.io/platypus/reference/platypus_spec.md).

## Details

`input_shape` is the tile the model sees: two numbers for images, three
for volumes. Its length is what makes a model 2D or 3D - there is no
separate switch. Every spatial dimension must divide by `2^blocks`, or
the decoder cannot line up with the encoder, and that is checked when
the specification is built rather than part-way through training.

## Examples

``` r
u_net("unet", input_shape = c(256, 256), blocks = 4, filters = 16)
#> $name
#> [1] "unet"
#> 
#> $architecture
#> [1] "u_net"
#> 
#> $input_shape
#> [1] 256 256
#> 
#> $channels
#> [1] 3
#> 
#> $n_class
#> [1] 2
#> 
#> $blocks
#> [1] 4
#> 
#> $filters
#> [1] 16
#> 
#> $block_width
#> [1] 2
#> 
#> $dropout
#> [1] 0
#> 
#> $batch_normalization
#> [1] TRUE
#> 
#> $separable_conv
#> [1] FALSE
#> 
#> $spatial_dropout
#> [1] TRUE
#> 
#> $upsample
#> [1] FALSE
#> 
#> $deep_supervision
#> [1] FALSE
#> 
#> $activation
#> [1] "relu"
#> 
#> $initialiser
#> [1] "he_normal"
#> 
#> $loss
#> $loss$name
#> [1] "cce"
#> 
#> $loss$label_smoothing
#> [1] 0
#> 
#> 
#> $metrics
#> $metrics[[1]]
#> $metrics[[1]]$name
#> [1] "iou"
#> 
#> $metrics[[1]]$smooth
#> [1] 1
#> 
#> $metrics[[1]]$include_background
#> [1] TRUE
#> 
#> 
#> 
#> $optimizer
#> $optimizer$name
#> [1] "adam"
#> 
#> $optimizer$learning_rate
#> [1] 0.001
#> 
#> $optimizer$beta_1
#> [1] 0.9
#> 
#> $optimizer$beta_2
#> [1] 0.999
#> 
#> $optimizer$eps
#> [1] 1e-08
#> 
#> $optimizer$weight_decay
#> [1] 0
#> 
#> $optimizer$amsgrad
#> [1] FALSE
#> 
#> 
#> $callbacks
#> list()
#> 
#> $augmentation
#> NULL
#> 
#> $epochs
#> [1] 10
#> 
#> $batch_size
#> [1] 8
#> 
#> $splits
#> NULL
#> 
#> $weights
#> NULL
#> 
#> $fit
#> [1] TRUE
#> 
#> $encoder
#> NULL
#> 
#> $pretrained
#> NULL
#> 
#> $freeze_encoder
#> NULL
#> 
#> $encoder_learning_rate
#> NULL
#> 

# A large image, tiled rather than shrunk
u_net("hd", input_shape = c(256, 256), splits = c(4, 6))
#> $name
#> [1] "hd"
#> 
#> $architecture
#> [1] "u_net"
#> 
#> $input_shape
#> [1] 256 256
#> 
#> $channels
#> [1] 3
#> 
#> $n_class
#> [1] 2
#> 
#> $blocks
#> [1] 4
#> 
#> $filters
#> [1] 16
#> 
#> $block_width
#> [1] 2
#> 
#> $dropout
#> [1] 0
#> 
#> $batch_normalization
#> [1] TRUE
#> 
#> $separable_conv
#> [1] FALSE
#> 
#> $spatial_dropout
#> [1] TRUE
#> 
#> $upsample
#> [1] FALSE
#> 
#> $deep_supervision
#> [1] FALSE
#> 
#> $activation
#> [1] "relu"
#> 
#> $initialiser
#> [1] "he_normal"
#> 
#> $loss
#> $loss$name
#> [1] "cce"
#> 
#> $loss$label_smoothing
#> [1] 0
#> 
#> 
#> $metrics
#> $metrics[[1]]
#> $metrics[[1]]$name
#> [1] "iou"
#> 
#> $metrics[[1]]$smooth
#> [1] 1
#> 
#> $metrics[[1]]$include_background
#> [1] TRUE
#> 
#> 
#> 
#> $optimizer
#> $optimizer$name
#> [1] "adam"
#> 
#> $optimizer$learning_rate
#> [1] 0.001
#> 
#> $optimizer$beta_1
#> [1] 0.9
#> 
#> $optimizer$beta_2
#> [1] 0.999
#> 
#> $optimizer$eps
#> [1] 1e-08
#> 
#> $optimizer$weight_decay
#> [1] 0
#> 
#> $optimizer$amsgrad
#> [1] FALSE
#> 
#> 
#> $callbacks
#> list()
#> 
#> $augmentation
#> NULL
#> 
#> $epochs
#> [1] 10
#> 
#> $batch_size
#> [1] 8
#> 
#> $splits
#> [1] 4 6
#> 
#> $weights
#> NULL
#> 
#> $fit
#> [1] TRUE
#> 
#> $encoder
#> NULL
#> 
#> $pretrained
#> NULL
#> 
#> $freeze_encoder
#> NULL
#> 
#> $encoder_learning_rate
#> NULL
#> 
```
