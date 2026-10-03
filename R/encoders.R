#' Pretrained encoders, and what they are worth
#'
#' `u_net()` and the other model constructors take an `encoder`, which replaces the
#' built-in contracting path with a backbone architecture, and a separate `pretrained`
#' flag that loads that backbone's ImageNet weights. This page is the measurement behind
#' them, because the honest summary is narrower than the idea.
#'
#' @section How to ask:
#'
#' ```r
#' u_net("polyps", input_shape = c(256, 256),
#'       encoder = "resnet34",      # the architecture
#'       pretrained = TRUE,         # and its ImageNet weights
#'       freeze_encoder = 5)        # held still while the decoder settles
#' ```
#'
#' Two flags rather than one, so naming an architecture never reaches the network on a
#' machine that has none - a hospital network often has none.
#'
#' @section What it is worth:
#'
#' Measured on three datasets, one per modality, three seeds each, every configuration
#' compared at its best validation Dice with the best weights restored. Training sets of
#' 40 to 100 images on purpose, because that is the regime transfer learning is for.
#'
#' | configuration | DS Bowl, microscopy | SIIM-ACR, chest X-ray | Kvasir-SEG, endoscopy |
#' | ------------- | ------------------- | --------------------- | --------------------- |
#' | built-in encoder, 1.9M parameters | **0.8460** | 0.0531 | 0.5830 |
#' | `encoder = "resnet34"`, 9.0M | 0.7891 | 0.1073 | 0.5670 |
#' | the same, `pretrained`, `freeze_encoder = 5` | 0.8333 | **0.1203** | **0.5932** |
#' | the same, `encoder_learning_rate = lr/10` | 0.8326 | 0.1011 | 0.5329 |
#'
#' Four things come out of it, and each repeats on all three:
#'
#' \describe{
#'   \item{Use `freeze_encoder`, not `encoder_learning_rate` alone}{Freezing was the
#'     better of the two everywhere. On Kvasir a tenth of the rate is the *worst* row in
#'     the table - below the built-in encoder and below the same backbone trained from
#'     nothing. Giving one rate to the whole network is worse still: a randomly
#'     initialised decoder's gradients undo the transferred weights before the decoder has
#'     learned anything worth having.}
#'   \item{Do not expect a better score}{The best pretrained configuration beats the
#'     built-in encoder on one dataset of three, by 0.0102, against a seed-to-seed spread
#'     of 0.0170 - inside the noise. And endoscopy is the fairest test available within
#'     medical imaging: camera photographs of tissue, three channels, natural lighting.
#'     With the backbone frozen for a whole run on DS Bowl the score is 0.6068 against
#'     0.8460, so the features genuinely do not transfer to these images.}
#'   \item{Expect to reach the same score sooner}{On Kvasir the built-in encoder needed
#'     150, 110 and 83 epochs across its three seeds; pretrained and frozen needed 64, 82
#'     and 83. That halving is the one benefit that showed up consistently, and on a
#'     dataset that takes a day to train it is the difference that matters.}
#'   \item{A bigger encoder earns its place where the target is thin}{On SIIM-ACR, where a
#'     pneumothorax covers a median 0.59% of the frame, every backbone variant roughly
#'     doubles the built-in encoder's score - and the built-in encoder's spread between
#'     seeds there (0.0586) is larger than its own mean, with two of three seeds never
#'     learning at all. This is a reason to be able to swap the encoder that has nothing
#'     to do with ImageNet.}
#' }
#'
#' Measure on your own images before trusting any of this: the regime where transfer helps
#' is narrow, and three public datasets are three public datasets. The engine ships
#' `examples/compare_pretrained_encoders.py`, which is this measurement parameterised.
#'
#' @section Which backbones:
#'
#' Any convolutional family timm names, as timm names it: `"resnet18"`, `"resnet34"`,
#' `"resnet50"`, `"resnext50_32x4d"`, `"efficientnet_b0"` through `"b3"`,
#' `"mobilenetv3_large_100"`, `"densenet121"`, `"regnety_040"`, `"vgg16"`,
#' `"inception_v3"` and others. `blocks` then says how many of the backbone's stages to
#' use.
#'
#' Two kinds are refused rather than approximated:
#'
#' - **A 3D `input_shape`.** ImageNet is images, so there is nothing to transfer to a
#'   volume. Refused while the specification is read, before anything is downloaded.
#' - **The patch-based families** (`"convnext_*"`, `"swin_*"`). Their features begin at a
#'   quarter of the input, so the half-resolution level a U-shaped decoder recovers fine
#'   detail from does not exist. Inventing it would put a made-up level exactly where the
#'   detail comes from.
#'
#' The finest level is always platypus's own: every ImageNet stem halves the input, so no
#' backbone has a full-resolution feature, and the decoder's output head sits on one. That
#' stage is trained from scratch, which is why `freeze_encoder` and
#' `encoder_learning_rate` leave it alone - it is the one part with everything to learn.
#'
#' @section Installing:
#'
#' Nothing to do. The environment this package builds always includes the backbones, which
#' cost 10 MB against PyTorch's own several hundred - not a trade worth making anyone opt
#' into. On a GTX 10-series card or older see [platypus_use_torch()].
#'
#' @seealso [models] for the arguments, [platypus_spec()] for the specification.
#' @name encoders
NULL
