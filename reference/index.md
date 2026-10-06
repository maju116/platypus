# Package index

## Setting up

The engine is Python, installed on first use. These three say where it
is, what it found, and which build of PyTorch to ask for.

- [`platypus_status()`](https://maju116.github.io/platypus/reference/platypus_status.md)
  : Is the engine ready?
- [`platypus_device()`](https://maju116.github.io/platypus/reference/platypus_device.md)
  : Where the work will happen
- [`platypus_use_torch()`](https://maju116.github.io/platypus/reference/platypus_use_torch.md)
  : Ask for a particular PyTorch build

## A specification

One object says what to train, on what data, with what. Built from
arguments or from YAML - the same specification either way.

- [`platypus_spec()`](https://maju116.github.io/platypus/reference/platypus_spec.md)
  : Build an experiment specification
- [`segmentation_data()`](https://maju116.github.io/platypus/reference/segmentation_data.md)
  : Where the images and masks are
- [`detection_data()`](https://maju116.github.io/platypus/reference/detection_data.md)
  : Where the images and their annotation files are
- [`u_net()`](https://maju116.github.io/platypus/reference/models.md)
  [`u_net_plus_plus()`](https://maju116.github.io/platypus/reference/models.md)
  [`res_u_net()`](https://maju116.github.io/platypus/reference/models.md)
  [`linknet()`](https://maju116.github.io/platypus/reference/models.md)
  : Segmentation models
- [`yolo3()`](https://maju116.github.io/platypus/reference/yolo3.md) :
  YOLOv3
- [`as.list(`*`<platypus_spec>`*`)`](https://maju116.github.io/platypus/reference/as.list.platypus_spec.md)
  : Turn a specification back into plain R data

## What a model is made of

Losses, metrics, optimisers, callbacks, augmentation, encoders.

- [`loss_iou()`](https://maju116.github.io/platypus/reference/losses.md)
  [`loss_dice()`](https://maju116.github.io/platypus/reference/losses.md)
  [`loss_cce()`](https://maju116.github.io/platypus/reference/losses.md)
  [`loss_cce_dice()`](https://maju116.github.io/platypus/reference/losses.md)
  [`loss_focal()`](https://maju116.github.io/platypus/reference/losses.md)
  [`loss_tversky()`](https://maju116.github.io/platypus/reference/losses.md)
  [`loss_focal_tversky()`](https://maju116.github.io/platypus/reference/losses.md)
  [`loss_combo()`](https://maju116.github.io/platypus/reference/losses.md)
  [`loss_lovasz()`](https://maju116.github.io/platypus/reference/losses.md)
  : Loss functions
- [`loss_boundary()`](https://maju116.github.io/platypus/reference/loss_boundary.md)
  : A loss that knows how far a wrong voxel is from the truth
- [`metric_iou()`](https://maju116.github.io/platypus/reference/metrics.md)
  [`metric_dice()`](https://maju116.github.io/platypus/reference/metrics.md)
  [`metric_tversky()`](https://maju116.github.io/platypus/reference/metrics.md)
  : Metrics
- [`optimizer_adam()`](https://maju116.github.io/platypus/reference/optimizers.md)
  [`optimizer_adamw()`](https://maju116.github.io/platypus/reference/optimizers.md)
  [`optimizer_sgd()`](https://maju116.github.io/platypus/reference/optimizers.md)
  [`optimizer_rmsprop()`](https://maju116.github.io/platypus/reference/optimizers.md)
  [`optimizer_adagrad()`](https://maju116.github.io/platypus/reference/optimizers.md)
  [`optimizer_adadelta()`](https://maju116.github.io/platypus/reference/optimizers.md)
  [`optimizer_adamax()`](https://maju116.github.io/platypus/reference/optimizers.md)
  [`optimizer_nadam()`](https://maju116.github.io/platypus/reference/optimizers.md)
  : Optimisers
- [`callback_early_stopping()`](https://maju116.github.io/platypus/reference/callbacks.md)
  [`callback_model_checkpoint()`](https://maju116.github.io/platypus/reference/callbacks.md)
  [`callback_reduce_lr_on_plateau()`](https://maju116.github.io/platypus/reference/callbacks.md)
  [`callback_cosine_annealing()`](https://maju116.github.io/platypus/reference/callbacks.md)
  [`callback_csv_logger()`](https://maju116.github.io/platypus/reference/callbacks.md)
  [`callback_terminate_on_nan()`](https://maju116.github.io/platypus/reference/callbacks.md)
  : Callbacks
- [`augment()`](https://maju116.github.io/platypus/reference/augment.md)
  : An augmentation step
- [`available_augmentations()`](https://maju116.github.io/platypus/reference/available_augmentations.md)
  : Which augmentations are available
- [`encoders`](https://maju116.github.io/platypus/reference/encoders.md)
  : Pretrained encoders, and what they are worth

## The data on disk

Splitting by patient rather than by slice, and looking at what is
actually there before spending an afternoon training on it.

- [`platypus_split()`](https://maju116.github.io/platypus/reference/platypus_split.md)
  : Split a dataset into training, validation and test sets
- [`split_files()`](https://maju116.github.io/platypus/reference/split_files.md)
  : The files in one part of a split
- [`read_images()`](https://maju116.github.io/platypus/reference/read_images.md)
  : Read images as the model sees them
- [`read_masks()`](https://maju116.github.io/platypus/reference/read_masks.md)
  : Read mask files as class indices
- [`volume_info()`](https://maju116.github.io/platypus/reference/volume_info.md)
  : What is in a volume file
- [`series_report()`](https://maju116.github.io/platypus/reference/series_report.md)
  : Check folders of DICOM slices before training on them
- [`mask_report()`](https://maju116.github.io/platypus/reference/mask_report.md)
  : What the colormap or labels match in your masks

## Training

- [`platypus_fit()`](https://maju116.github.io/platypus/reference/platypus_fit.md)
  : Train the models in a specification
- [`training_history()`](https://maju116.github.io/platypus/reference/training_history.md)
  : The epoch-by-epoch record
- [`plot(`*`<platypus_fit>`*`)`](https://maju116.github.io/platypus/reference/plot.platypus_fit.md)
  : Plot what happened during training
- [`save_weights()`](https://maju116.github.io/platypus/reference/save_weights.md)
  : Save a trained model's weights
- [`available_weights()`](https://maju116.github.io/platypus/reference/available_weights.md)
  : The published weights, and what they are

## Predicting and scoring

- [`predict(`*`<platypus_fit>`*`)`](https://maju116.github.io/platypus/reference/predict.platypus_fit.md)
  : Predict masks
- [`evaluate()`](https://maju116.github.io/platypus/reference/evaluate.md)
  : Compare the trained models
- [`evaluate_cases()`](https://maju116.github.io/platypus/reference/evaluate_cases.md)
  [`summary(`*`<platypus_cases>`*`)`](https://maju116.github.io/platypus/reference/evaluate_cases.md)
  : Score every case separately
- [`evaluate_classes()`](https://maju116.github.io/platypus/reference/evaluate_classes.md)
  : Average precision per class
- [`evaluate_images()`](https://maju116.github.io/platypus/reference/evaluate_images.md)
  : Score every image separately
- [`detection_anchors()`](https://maju116.github.io/platypus/reference/detection_anchors.md)
  : The anchors a detector used, and whether they were fitted
- [`detection_crops()`](https://maju116.github.io/platypus/reference/detection_crops.md)
  : Cut every detection out of the image it was found in

## Looking at the results

A figure is the point. These are the ones you put in a paper.

- [`plot_masks()`](https://maju116.github.io/platypus/reference/plot_masks.md)
  : Look at masks beside the images they came from
- [`plot_boxes()`](https://maju116.github.io/platypus/reference/plot_boxes.md)
  : Draw boxes on images
- [`plot_anchors()`](https://maju116.github.io/platypus/reference/plot_anchors.md)
  : Draw the anchors against the boxes they were fitted to
- [`overlay_mask()`](https://maju116.github.io/platypus/reference/overlay_mask.md)
  : Lay a mask over the image it belongs to
- [`overlay_agreement()`](https://maju116.github.io/platypus/reference/overlay_agreement.md)
  : Where a prediction and the truth disagree

## Masks and volumes

- [`mask_classes()`](https://maju116.github.io/platypus/reference/mask_classes.md)
  : Read a colour mask back to class indices
- [`mask_colours()`](https://maju116.github.io/platypus/reference/mask_colours.md)
  : Paint a mask with its colormap
- [`mask_coverage()`](https://maju116.github.io/platypus/reference/mask_coverage.md)
  : How much of each class a mask covers
- [`mask_volume()`](https://maju116.github.io/platypus/reference/mask_volume.md)
  : How much of a volume a class occupies, in millilitres
- [`unite_masks()`](https://maju116.github.io/platypus/reference/unite_masks.md)
  : Combine several binary masks into one
- [`binary_colormap`](https://maju116.github.io/platypus/reference/colormaps.md)
  [`binary_labels`](https://maju116.github.io/platypus/reference/colormaps.md)
  [`voc_colormap`](https://maju116.github.io/platypus/reference/colormaps.md)
  [`voc_labels`](https://maju116.github.io/platypus/reference/colormaps.md)
  : Colormaps and their labels
- [`ct_windows()`](https://maju116.github.io/platypus/reference/ct_windows.md)
  : The named CT windows
- [`save_masks()`](https://maju116.github.io/platypus/reference/save_masks.md)
  : Write masks to files
- [`save_volumes()`](https://maju116.github.io/platypus/reference/save_volumes.md)
  : Save predicted volumes as NIfTI
