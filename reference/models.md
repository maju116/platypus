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
  channels = NULL,
  n_class = NULL,
  blocks = NULL,
  filters = NULL,
  block_width = NULL,
  dropout = NULL,
  batch_normalization = NULL,
  separable_conv = NULL,
  spatial_dropout = NULL,
  upsample = NULL,
  deep_supervision = NULL,
  activation = NULL,
  initialiser = NULL,
  loss = NULL,
  metrics = NULL,
  optimizer = NULL,
  callbacks = NULL,
  augmentation = NULL,
  epochs = NULL,
  batch_size = NULL,
  splits = NULL,
  weights = NULL,
  fit = NULL,
  encoder = NULL,
  pretrained = NULL,
  freeze_encoder = NULL,
  encoder_learning_rate = NULL
)

u_net_plus_plus(
  name,
  input_shape,
  channels = NULL,
  n_class = NULL,
  blocks = NULL,
  filters = NULL,
  block_width = NULL,
  dropout = NULL,
  batch_normalization = NULL,
  separable_conv = NULL,
  spatial_dropout = NULL,
  upsample = NULL,
  deep_supervision = NULL,
  activation = NULL,
  initialiser = NULL,
  loss = NULL,
  metrics = NULL,
  optimizer = NULL,
  callbacks = NULL,
  augmentation = NULL,
  epochs = NULL,
  batch_size = NULL,
  splits = NULL,
  weights = NULL,
  fit = NULL,
  encoder = NULL,
  pretrained = NULL,
  freeze_encoder = NULL,
  encoder_learning_rate = NULL
)

res_u_net(
  name,
  input_shape,
  channels = NULL,
  n_class = NULL,
  blocks = NULL,
  filters = NULL,
  block_width = NULL,
  dropout = NULL,
  batch_normalization = NULL,
  separable_conv = NULL,
  spatial_dropout = NULL,
  upsample = NULL,
  deep_supervision = NULL,
  activation = NULL,
  initialiser = NULL,
  loss = NULL,
  metrics = NULL,
  optimizer = NULL,
  callbacks = NULL,
  augmentation = NULL,
  epochs = NULL,
  batch_size = NULL,
  splits = NULL,
  weights = NULL,
  fit = NULL,
  encoder = NULL,
  pretrained = NULL,
  freeze_encoder = NULL,
  encoder_learning_rate = NULL
)

linknet(
  name,
  input_shape,
  channels = NULL,
  n_class = NULL,
  blocks = NULL,
  filters = NULL,
  block_width = NULL,
  dropout = NULL,
  batch_normalization = NULL,
  separable_conv = NULL,
  spatial_dropout = NULL,
  upsample = NULL,
  deep_supervision = NULL,
  activation = NULL,
  initialiser = NULL,
  loss = NULL,
  metrics = NULL,
  optimizer = NULL,
  callbacks = NULL,
  augmentation = NULL,
  epochs = NULL,
  batch_size = NULL,
  splits = NULL,
  weights = NULL,
  fit = NULL,
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

  Input channels. 3 for colour, 1 for greyscale. Derived from
  `channels_from` when
  [`segmentation_data()`](https://maju116.github.io/platypus/reference/segmentation_data.md)
  names one file per channel.

- n_class:

  **Removed.** The number of classes comes from the data - `colormap` or
  `labels` in
  [`segmentation_data()`](https://maju116.github.io/platypus/reference/segmentation_data.md) -
  and a model that also declared it could only ever disagree with its
  own data. Passing it is an error that says so.

- blocks:

  How many times the resolution is halved. Taken from the weights file
  when `weights` is given and this is not.

- filters:

  Filters in the first block; doubled at each level. Taken from the
  weights file when `weights` is given and this is not.

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
  [`augmentation_step()`](https://maju116.github.io/platypus/reference/augmentation_step.md)
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

## Where the defaults live

Every argument below except `name` and `input_shape` is `NULL` unless
you set it, and a `NULL` is left out of the request entirely - so the
engine's own default applies. The defaults are therefore written down
once, in `pyplatypus`, rather than here as well:


    channels 3        blocks 4          filters 16      block_width 2
    dropout 0         batch_normalization TRUE          separable_conv FALSE
    spatial_dropout TRUE                upsample FALSE  deep_supervision FALSE
    activation "relu"                   initialiser "he_normal"
    loss cce          metrics iou       optimizer adam  callbacks none
    epochs 10         batch_size 8      fit TRUE

The second reason matters more than tidiness. A value sent is
indistinguishable from a value chosen, so defaults travelling from here
would silence the engine's own knowledge: `weights` naming a published
file records its architecture, `blocks` and `filters`, and those are
adopted **only** where the specification stayed silent. Sending
`blocks = 4` because that is what R happened to say would override a
file that knows better.

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
#> $blocks
#> [1] 4
#> 
#> $filters
#> [1] 16
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
#> $splits
#> [1] 4 6
#> 
```
