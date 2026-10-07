# Change Log

All notable changes to this package will be documented in this file. Two things about its
shape are worth stating, because neither is the usual one.

**There is no release yet.** `remotes::install_github("maju116/platypus")` serves the
default branch, so "the version I had working" is a commit rather than a tag. Entries are
therefore dated rather than numbered: the heading they sit under is the version in
`DESCRIPTION`, and the `0.1.1` below it is the 2020 package this one replaced.

**Every entry names the engine it needs.** The computation lives in
[pyplatypus](https://github.com/maju116/pyplatypus) and this package pins one exact version
of it. A feature lands in the engine, the engine is published to PyPI, and only then can
this package reach it - so the pin moving is part of a change here, not housekeeping
alongside it. The pin is what stops a release of this package breaking because the engine
drifted underneath it.

## [0.2.0] - unreleased

A different package from 0.1.1 under the same name: that one built Keras models through the
R `keras` package and handed them back, this one describes a whole run as a specification
and computes it in PyTorch through pyplatypus. See **Coming from platypus 0.1.1** in the
README for what moved and what is gone.

### 2026-10-07 - engine 0.6.0a2, then 0.6.0a4, then 0.7.0a1

#### Changed, and it breaks code

 - **Eight functions were renamed.** The engine and this package had grown two words for the
   same thing in eight places, and a reader moving between the halves paid for every one.
   One name per concept now, and where the two disagreed **the engine's name won**, because
   it is the layer that cannot be renamed later without changing the configuration format
   and the schema. The exception is `available_transforms()`, where this package already had
   a family of `available_*` names and the engine followed instead.

   | was | is |
   |---|---|
   | `platypus_split()` | `split_dataset()` |
   | `split_files()` | `split_path()` |
   | `mask_report()` | `inspect_masks()` |
   | `mask_classes()` | `colours_to_classes()` |
   | `mask_colours()` | `classes_to_colours()` |
   | `save_weights()` | `export_weights()` |
   | `available_augmentations()` | `available_transforms()` |
   | `augment()` | `augmentation_step()` |

   No aliases: two names for one thing is what this removes. The S3 **class** returned by
   `split_dataset()` is still `platypus_split` - that is the object's identity, and every
   class in the package is prefixed.

 - `PYPLATYPUS_VERSION` is `pyplatypus_version`. It was the only ALL-CAPS name in the
   package, so keeping it would have meant a lint exception for a style used exactly once.

#### Added

 - **`callback_swa()`**: average the weights over the last part of the run. What it is worth
   is not in the score column - Dice and per-case error do not move - it is in the spread.
   Measured over three seeds, the volume bias went from `+4.5% / +12.2% / -11.6%` to
   `-0.1% / +4.2% / +2.1%`, 5.7 times tighter. The help page says so in those words, because
   somebody reading "weight averaging" will look at Dice, find nothing, and conclude wrongly.
   Refused beside `callback_cosine_annealing()`: a cosine has decayed to its floor by the
   time averaging begins, so the average is the last epoch.

 - **`segmentation_data(validation = FALSE)`**: say a run has no validation set. The third
   answer to the question a path and a split already answered, and it had to be a stated
   value rather than an absence, because "I have no validation set" and "I forgot" look
   identical and only one of them is a decision. It went in the same argument rather than a
   new one, so nothing can contradict it.

#### Fixed

 - The Python shim shipped inside this package is now formatted and linted by ruff in CI,
   beside lintr. It had never been.

### 2026-10-07 - the documentation site

 - **The site is Quarto, not pkgdown**, generated from `tools/reference-groups.yml` into a
   sidebar nested in eight groups, and themed from a `_brand.yml` that is byte-identical to
   pyplatypus's - so the two sites look alike by construction rather than by anybody
   remembering to copy a colour.
 - **Every old address still works.** 105 redirect stubs cover all 99 addresses the pkgdown
   site published, the eight renames among them.

### 2026-10-06 - engine 0.5.0a1 through 0.6.0a1

#### Added

 - **`segmentation_data(split = )`** and `detection_data(split = )`: divide one training
   folder into splits without writing any files. The argument is a list of `fractions` (two
   numbers, or three to cut a test set too) and `group_by`, which is a **required key that
   may be `NULL`**. The mistake it prevents leaves no trace: slices of one patient on both
   sides make validation measure memory rather than generalisation, and nothing in the output
   says so. An omitted key would be that choice made silently; `NULL` is the same choice made
   on purpose. For detection this is the only way to divide one folder, because
   `split_dataset()` writes a `masks` column.
 - **`detection_crops()`**: cut the detected objects out of the images they were found in, at
   a fixed size, for handing to a classifier. Crops come from the source image through the
   same reader the dataset uses, are rounded outward so the whole box survives, and
   letterbox rather than stretch.
 - **`yolo3(box_loss = "giou")`**. Measured on BCCD over three seeds, 150 epochs, it is
   *worse* on the metric it targets - matched overlap 0.7929 against 0.8038, established at
   2.7 standard errors - and its spread is sixty times wider. It ships because its
   coordinate term reads zero when the boxes are right, where the default term cannot reach
   zero however right they are. `?yolo3` leads with that table, because the name invites
   choosing it and the measured answer is the other way.
 - **`loss_boundary()`**: a region loss plus the signed distance to the truth's boundary.
   `region` and `alpha` default to `NULL` here and are left out of the request, so the
   engine's own defaults apply - `alpha = 0.9`, which it pins with a test, because at 0.5 the
   surface term owns half the objective while the prediction is still random and the volume
   bias spread goes to +/-28%. What the loss trades is accuracy per case for being unbiased
   over a series: measured over three seeds, the systematic volume bias went from -5.18% to
   -0.06% and the per-case absolute error from 16.90% to 22.17%.

#### Fixed

 - Every help page has an `\examples` section but one - 47 of 48 at the time, `encoders`
   being a concept page with no `\usage` - and a test parses every example in every `.Rd`,
   including inside `\dontrun{}`, and fails if it names a function that does not exist. What
   was at risk was never the functionality; it is the example *text* drifting from the API.
 - lintr runs over the package in CI and gates on its findings. Of 906 findings at lintr's
   defaults, 890 were the 80-character line limit this package does not use; at 100
   characters and `lint_package()`, 25 - two of which were defects rather than style.

### 2026-10-05 - engine 0.3.0a15, then 0.4.0a1

 - **`evaluate_images()`**: one row per image for a detector - `n_truth`, `n_predicted`,
   `matched`, `missed`, `spurious`, `mean_matched_iou`. Order by `missed` for the frames it
   cannot see, by `mean_matched_iou` for the ones where it finds everything and places it
   badly. No average precision per image: AP is the area under a ranking's curve and on one
   picture it moves on a single box.
 - **`platypus_spec(task = )`** can be stated, and is checked rather than trusted - it can
   confirm or refuse, never override. Unset, it is still derived from the constructors.
   Required in YAML and optional here is not an inconsistency: `task` is needed exactly where
   the type is not already stated, and `segmentation_data()` with `u_net()` can only mean one
   thing.

### 2026-10-04 - engine 0.3.0a12, then 0.3.0a13

 - **Object detection, from R.** `detection_data()`, `yolo3()`, `detection_anchors()`,
   `plot_boxes()`, and `predict()` returning a list of boxes in each image's own pixels -
   not an array, because images differ in size, so does the number of boxes, and a box in a
   letterboxed frame cannot be drawn on the photograph it came from.
 - **`plot_anchors()`**: the fitted anchors against the boxes they were fitted to. It is in
   the vignette twice, on the split they were fitted on and the one they were not.
 - A README figure for detection, drawn by the published `bccd-yolo3` with no training, and
   `tools/readme_figures.R` that reproduces it.
 - **Corrected**: the claim that COCO's anchors cover BCCD's boxes at 0.67 where fitted ones
   reach 0.92. Re-measured on the 2,805 real training boxes, the honest pair is **0.65
   against 0.88**. The direction survives and is large; both numbers were wrong in opposite
   directions and the gap was right by coincidence.

### 2026-10-03 - engine 0.3.0a10, then 0.3.0a11

 - **`u_net(encoder = "resnet34", pretrained = TRUE)`**, with `freeze_encoder` and
   `encoder_learning_rate`. `pretrained` is a separate explicit flag so that naming a
   backbone never downloads anything nobody asked for. Measured on three datasets,
   pretraining never clearly beats the built-in encoder; what it buys is epochs, and what a
   swappable encoder buys is capacity where the target is thin. The help page carries the
   table.
 - **`inspect_masks()`** and `platypus_fit(check_masks = FALSE)`: a declared class appearing
   in no mask has a target channel that is zero everywhere, so the model is never shown what
   it is being asked to find, and `platypus_fit()` now refuses on it. A threshold on the
   unmatched fraction cannot do this job - on a 0.6% lesion a wholly wrong colormap is
   quieter than JPEG artefacts in healthy data.
 - **Every help page renders its markdown.** `DESCRIPTION` had no
   `Roxygen: list(markdown = TRUE)` although every page had been written in markdown since
   the rewrite began, so not one cross-reference was a link: 0 to 42 links, 0 to 266 inline
   code spans, 36 of 40 pages changed. Nothing in the sources was wrong; nothing was
   converting them.

### 2026-09-27 - engine 0.3.0a6, then 0.3.0a8

 - **`predict(space = "source")`**: map a prediction back out of the model's grid onto the
   grid of the scan it came from, undoing the resampling and the crop. Returns a list, and
   the type change is the API being honest - scans differ in size, and quietly resizing them
   to match is how a mask ends up describing anatomy it was not computed from.
 - **Named weights.** `weights = "dsbowl-unet"` from the registry, `hf://owner/repo/file@commit`,
   or a local path, plus `export_weights()` for your own. A registry name carries a commit
   and the `hf://` form requires one, because weights that change under a stable name are the
   worst kind of irreproducibility.
 - The Data Science Bowl vignette now opens with a mask in four lines - published weights,
   `fit = FALSE`, `plot_masks()` - before any mention of training. First contact should not
   need a GPU and an afternoon.
 - **Fixed, and found by writing that vignette:** `weights = "dsbowl-unet"` could not work
   from R at all. The bridge asked for `pyplatypus` with no extras, so `huggingface_hub` was
   never in the environment reticulate builds. Nothing in either test suite could have caught
   it - the engine's tests install the extra deliberately and the R tests ran where it was
   already present.
 - The README says what changed since 0.1.1, written by diffing the two `NAMESPACE` files
   rather than from memory: of 0.1.1's 37 exports, 30 are gone, and three of the seven
   survivors are worse than gone because they are still callable and mean something else.

### 2026-09-26 - engine 0.2.0a2 through 0.3.0a5

The day the package stopped being a bridge and became a surface. Five engine releases, each
followed here.

 - **A worked vignette** on the Data Science Bowl, precomputed so that reading it needs no
   GPU, and a README.
 - **Continuous integration**: five `R CMD check` jobs - Ubuntu on R devel, release and
   oldrel-1, plus macOS and Windows on release - and an `engine` job that installs the built
   package and runs the suite against the engine resolved from PyPI. That last one is the
   job that matters: `pkgload::load_all()` exposes a package's internals, `library()` does
   not, and local work had been using the most permissive of the three ways a test can
   reach them.
 - **Plotting and masks**: `plot_masks()`, `plot_boxes()`, `overlay_mask()`,
   `overlay_agreement()`, `read_masks()`, `unite_masks()`, `mask_coverage()`.
 - **DICOM**, properly: modality LUT, a fixed window, MONOCHROME1 inverted. Plus
   `series_report()` for a folder of slices, whose decision is that **a problem is a column,
   not an exception** - a hundred cases contain a few bad ones, and stopping at the first
   turns an afternoon into a week.
 - **Splitting by patient** (`split_dataset()`, `split_path()`) and `evaluate_cases()`.
 - **Volumes.** `input_shape = c(64, 64, 32)` had been building a 3D U-Net since the first
   release; what was missing was reading NIfTI, where orientation is the whole problem, and
   `labels` as a second way to describe a mask, because volumes label with integers.
   `save_volumes()` writes NIfTI **carrying the affine of the scan the mask was predicted
   from** - without it a mask cannot be laid over its scan by any viewer, and the file looks
   perfectly normal while being wrong. `mask_volume()` reports millilitres, because a voxel
   count is meaningless outside one scanner.
 - **`target_spacing`**: resample to a common voxel size. What makes it necessary is not
   slice thickness but field of view - forty slices of 1 mm is 40 mm of patient and forty of
   2.5 mm is 100 mm, which is the ordinary state of clinical data.
 - **`channels_from`**: one channel per file, stating the order and never inferring it. A
   model trained with FLAIR in channel one and used where channel one is T1 answers
   plausibly, with the right shape and range, and nothing downstream can tell.
 - **Augmentation in 3D**, and `available_transforms(rank = 3)` to say what is usable - 97 of
   albumentations' transforms accept a volume and the rest raise from inside the library.
 - A second vignette, on volumes, which generates its own data. It ends on a model at **Dice
   0.88 whose volumes are 12-23% too large**, which is the argument for reporting volume
   beside overlap: Dice is dominated by an object's interior and forgiving about its
   boundary.
 - `segmentation_data(window = )` replaced `dicom_window`, since NIfTI needs the same
   thing. The old name is still accepted; giving both is an error.

### 2026-09-25 - engine 0.2.0a1

The package exists.

 - **The bridge.** `py_require()` plus uv, resolving an isolated ephemeral environment at
   `library(platypus)` - deliberately not the shared-Python arrangement that broke this
   package's Keras stack in 2020. `platypus_status()` says what it found rather than leaving
   a failed import to be guessed at, and `PLATYPUS_ENGINE_PATH` points the bridge at a source
   tree during development.
 - **`platypus_spec()`**, built either from arguments or from a YAML file and producing the
   same thing. One pipeline, two constructors.
 - **`platypus_fit()`**, `evaluate()`, `predict()`, `training_history()`, and the engine's
   complaints translated into R conditions rather than passed through as Python tracebacks.

## [0.1.1] - 2020-10-17

The last release of the TensorFlow package, kept here for the record rather than as an
ancestor of the entries above. It built Keras models through the R `keras` package: `yolo3()`
and `darknet53()` with `load_darknet_weights()` for the original Darknet weights, a plain
`u_net()`, `segmentation_generator()`, `loss_dice()`, the COCO and VOC colormaps, and
readers for LabelMe JSON and Pascal VOC XML.

It is still installable and still pinned:

```r
remotes::install_github("maju116/platypus@0.1.1")
```

Its source is on the `master` branch and the issues filed against it stay open. What has
aged is its dependencies rather than its code - the pinned TensorFlow no longer installs on
a current Python, which is the reason for a rewrite rather than a consequence of one.
