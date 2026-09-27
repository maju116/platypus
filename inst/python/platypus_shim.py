"""The R-facing edge of the engine.

Exceptions do not survive the crossing usefully: reticulate hands R the message as text
and drops the object, so the structured problems pyplatypus reports would be lost and the
R side would be left parsing its own prose. Everything here therefore returns data.

This lives in the R package rather than in pyplatypus because it exists for R's benefit.
pyplatypus raises exceptions, like any Python library should.
"""

from __future__ import annotations

from typing import Any


def _failure(error: Any) -> dict:
    payload = error.to_dict() if hasattr(error, "to_dict") else {}
    payload.setdefault("kind", type(error).__name__)
    payload.setdefault("message", str(error))
    payload.setdefault("problems", [])
    payload["ok"] = False
    return payload


def build_spec(config: dict, check_paths: bool = True) -> dict:
    """A validated spec, or the reasons it is not one."""
    import pyplatypus

    try:
        return {"ok": True, "spec": pyplatypus.from_dict(config, check_paths=check_paths)}
    except pyplatypus.PlatypusError as error:
        return _failure(error)


def load_spec(path: str, check_paths: bool = True) -> dict:
    """The same, from a YAML file. The same object comes out either way."""
    import pyplatypus

    try:
        return {"ok": True, "spec": pyplatypus.from_yaml(path, check_paths=check_paths)}
    except pyplatypus.PlatypusError as error:
        return _failure(error)


def spec_as_dict(spec: Any) -> dict:
    """Plain data, so R can look at a spec without asking Python about every field."""
    return spec.to_dict()


def _engine_failure(error: Any) -> dict:
    """A failure during a run, rather than in a specification.

    Caught broadly on purpose. A specification error is ours and arrives well described;
    a run can fail on anything underneath - the GPU running out of memory, a file that
    vanished, a driver that cannot do what the build expects - and those exceptions come
    from libraries the user never invoked. Letting them through as a reticulate traceback
    would be the least useful moment possible to stop being readable, because it happens
    after the expensive part. The type is kept in the message so it stays diagnosable.
    """
    return {
        "ok": False,
        "kind": "engine_error",
        "message": f"{type(error).__name__}: {error}",
        "problems": [],
    }


def build_engine(spec: Any, device: str | None = None, num_workers: int = 0,
                 strict_data: bool = True) -> dict:
    import pyplatypus

    try:
        engine = pyplatypus.Engine(
            spec, device=device, num_workers=int(num_workers), strict_data=strict_data
        )
    except pyplatypus.PlatypusError as error:
        return _failure(error)
    except Exception as error:  # noqa: BLE001 - see _engine_failure
        return _engine_failure(error)
    return {"ok": True, "engine": engine}


def run_fit(engine: Any, verbose: bool = False) -> dict:
    """Train every model, and hand back the history as rows rather than objects."""
    import pyplatypus

    try:
        histories = engine.fit(verbose=bool(verbose))
    except pyplatypus.PlatypusError as error:
        return _failure(error)
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)

    rows, reasons = [], {}
    for name, history in histories.items():
        for record in history.records:
            rows.append({"model": name, **record})
        if history.stop_reason:
            reasons[name] = history.stop_reason
    return {"ok": True, "history": rows, "stop_reasons": reasons,
            "models": list(histories)}


def evaluation_table(engine: Any, split: str = "validation") -> dict:
    import pyplatypus

    try:
        return {"ok": True, "table": engine.evaluate(split)}
    except pyplatypus.PlatypusError as error:
        return _failure(error)
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)


def predictions(engine: Any, model_name: str, split: str = "test",
                as_class: bool = True, space: str = "model") -> dict:
    """Masks for a split.

    `as_class` collapses the channel axis to the class index, which is the mask someone
    actually wants to look at; the probabilities are there for anyone who needs them.

    `space` is `"model"` or `"source"`. In source space the engine returns one array per scan,
    on that scan's own grid, so the result is a list - which is why the answer says which space
    it is in rather than leaving R to infer it from the shape.
    """
    import numpy as np
    import pyplatypus

    try:
        probabilities = engine.predict(model_name, split=split, space=space)
    except pyplatypus.PlatypusError as error:
        return _failure(error)
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)

    def to_class(array):
        # +1 so the classes are 1-based on arrival: R indexes from one, and a mask whose
        # background is 0 while its colormap starts at 1 is a trap laid for later.
        return (array.argmax(axis=-1) + 1).astype(np.int32)

    if isinstance(probabilities, list):
        # Source space: one array per scan, each on its own grid, so this cannot become one
        # array and R receives a list. Keeping the shapes apart is the point - see
        # `?predict.platypus_fit`.
        masks = [to_class(one) if as_class else one for one in probabilities]
        return {"ok": True, "masks": masks, "type": "class" if as_class else "probability",
                "space": "source"}

    masks = to_class(probabilities) if as_class else probabilities
    return {"ok": True, "masks": masks, "type": "class" if as_class else "probability",
            "space": "model"}


def model_names(engine: Any) -> list:
    return list(engine.runs)


def device_report() -> dict:
    """Where the work will actually happen, and whether that is what was intended."""
    import torch

    cuda = torch.cuda.is_available()
    report = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda or "cpu-only build",
        "cuda_available": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "arch_list": list(torch.cuda.get_arch_list()) if cuda else [],
    }
    if not cuda:
        # A card that torch cannot use is worth saying out loud: the run falls back to the
        # processor and takes perhaps ten times as long, which otherwise looks like
        # nothing at all going wrong.
        #
        # device_count() is the signal, not is_available(). With a CUDA 13 build on a
        # Pascal card it returns 1 - torch can enumerate the card, it just cannot
        # initialise it - which is precisely the case worth reporting. Testing for zero
        # here, as this first did, detects nothing.
        try:
            count = torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            count = 0
        report["gpu_present_but_unusable"] = count > 0 or _nvidia_smi_sees_a_gpu()
    return report


def _nvidia_smi_sees_a_gpu() -> bool:
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def read_images(paths: list, channels: int = 3, size: list | None = None,
                nearest: bool = False) -> dict:
    """Read images exactly as the data pipeline would.

    Not a convenience: a picture plotted from a differently resized copy is a picture of
    something the model never saw, and the disagreements it shows may be the resizing.
    """
    import numpy as np
    from pyplatypus.data import read_image

    try:
        arrays = [
            read_image(p, channels=int(channels),
                       size=tuple(int(s) for s in size) if size else None,
                       nearest=bool(nearest))
            for p in paths
        ]
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)

    shapes = {a.shape for a in arrays}
    if len(shapes) > 1 and size is None:
        return {
            "ok": False, "kind": "image_error", "problems": [],
            "message": (
                "these images are not all the same size "
                f"({', '.join('x'.join(map(str, s)) for s in sorted(shapes))}), so they "
                "cannot go into one array. Pass `size` to read them at a common size."
            ),
        }
    return {"ok": True, "images": np.stack(arrays)}


def write_masks(masks, paths: list) -> dict:
    """Write coloured masks to the given files.

    Through the engine rather than through an R image library, for the same reason as
    reading: one decoder, one set of conventions, and no extra dependency on the R side
    for something the engine can already do.
    """
    import numpy as np
    import pathlib
    from PIL import Image

    try:
        arrays = np.asarray(masks, dtype=np.uint8)
        if arrays.ndim == 3:
            arrays = arrays[None]
        if len(arrays) != len(paths):
            return {
                "ok": False, "kind": "image_error", "problems": [],
                "message": f"{len(arrays)} masks but {len(paths)} paths",
            }
        for array, path in zip(arrays, paths):
            target = pathlib.Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).convert("RGB").save(target)
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)
    return {"ok": True, "paths": [str(p) for p in paths]}


def window_presets() -> dict:
    """The named CT windows, so R does not keep its own copy to drift out of step."""
    from pyplatypus.spec.common import WINDOWS

    return {name: list(pair) for name, pair in WINDOWS.items()}


def split_data(root: str, out_dir: str, mode: str = "nested_dirs",
               subdirs: list | None = None, column_sep: str = ";",
               fractions: list | None = None, group_by: str | None = None,
               seed: int = 0, strict: bool = True, relative: bool = True) -> dict:
    """Divide one directory into train/validation/test CSVs.

    `group_by` crosses over as written, on purpose. It is a Python regular expression, and
    translating patterns between R and Python in the background is a silent failure waiting
    to happen: the two dialects agree often enough to lull, and differ exactly where it
    matters. Documented in ?platypus_split instead.
    """
    import pyplatypus

    try:
        report = pyplatypus.split_dataset(
            root, out_dir,
            mode=mode,
            subdirs=tuple(subdirs) if subdirs else ("images", "masks"),
            column_sep=column_sep,
            fractions=tuple(float(f) for f in fractions) if fractions else (0.7, 0.15, 0.15),
            group_by=group_by,
            seed=int(seed),
            strict=bool(strict),
            relative=bool(relative),
        )
    except pyplatypus.PlatypusError as error:
        return _failure(error)
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)
    return {"ok": True, **report}


def case_table(engine: Any, model_name: str, split: str = "validation",
               group_by: str | None = None) -> dict:
    """One row per case - or per group, with `group_by` - instead of one per model."""
    import pyplatypus

    try:
        rows = engine.evaluate_cases(model_name, split=split, group_by=group_by)
    except pyplatypus.PlatypusError as error:
        return _failure(error)
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)
    return {"ok": True, "table": rows, "label": "group" if group_by else "case"}


def case_summary(rows: list) -> dict:
    """The distribution of the per-case scores.

    Computed by the engine rather than in R although it is only means and quantiles: two
    implementations of the same summary would eventually disagree, and a table that differs
    between the R and Python surfaces is worse than no table.
    """
    import pyplatypus

    try:
        return {"ok": True, "table": pyplatypus.summarise_cases(list(rows))}
    except pyplatypus.PlatypusError as error:
        return _failure(error)
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)


def volume_support() -> bool:
    """Whether the engine in use can read volumes at all."""
    try:
        from pyplatypus.data import volumes  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def volume_info(paths: list) -> dict:
    """Shape and voxel spacing for each volume, in the canonical order the reader uses.

    Spacing is not decoration. A segmented lesion counted in voxels is a number that means
    nothing outside the scanner it came from; the same count times the voxel volume is
    millilitres, which is what a report says and what a clinician compares.
    """
    from pyplatypus.data.volumes import VolumeError, read_volume, volume_spacing

    rows = []
    try:
        for path in paths:
            spacing = volume_spacing(path)
            shape = read_volume(path, nearest=True).shape
            rows.append({
                "path": str(path),
                "spacing": list(spacing),
                "shape": list(shape[:3]),
                "voxel_ml": float(spacing[0] * spacing[1] * spacing[2] / 1000.0),
            })
    except VolumeError as error:
        return _failure(error)
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)
    return {"ok": True, "volumes": rows}


def write_volumes(masks, paths: list, reference: list) -> dict:
    """Write label-map volumes as NIfTI, carrying each reference scan's affine.

    The affine is the point. A mask array on its own cannot be laid over the scan it came
    from: every viewer, every registration tool and every volume calculation reads position
    and spacing out of the affine, and a file without the right one is either silently
    misplaced or rejected. So the mask is written with the geometry of the scan it was
    predicted from, which also means it opens on top of it in any viewer with no further
    work.

    Written as integer labels rather than as colours, because that is what a volume format
    holds and what a segmentation tool expects to read back.
    """
    import pathlib

    import numpy as np

    try:
        import nibabel as nib

        arrays = np.asarray(masks)
        if arrays.ndim == 3:
            arrays = arrays[None]
        if len(arrays) != len(paths):
            return {"ok": False, "kind": "volume_error", "problems": [],
                    "message": f"{len(arrays)} masks but {len(paths)} paths"}
        if len(reference) != len(paths):
            return {"ok": False, "kind": "volume_error", "problems": [],
                    "message": (f"{len(reference)} reference volumes but {len(paths)} "
                                "masks; each mask needs the scan it was predicted from")}

        written = []
        for array, path, source in zip(arrays, paths, reference):
            source_image = nib.as_closest_canonical(nib.load(str(source)))
            target = pathlib.Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)

            labels = np.asarray(array)
            if labels.shape != source_image.shape[:3]:
                return {
                    "ok": False, "kind": "volume_error", "problems": [],
                    "message": (
                        f"mask {labels.shape} does not match '{source}' "
                        f"{tuple(source_image.shape[:3])}. A mask written with the wrong "
                        "geometry lands in the wrong place, which is worse than failing."
                    ),
                }
            image = nib.Nifti1Image(labels.astype(np.int16), source_image.affine,
                                    dtype=np.int16)
            nib.save(image, str(target))
            written.append(str(target))
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)
    return {"ok": True, "paths": written}


def series_report(paths: list) -> dict:
    """Check each directory of DICOM slices and report what is wrong with it.

    One row per directory, and a problem is a value in a column rather than an exception. A
    hundred cases out of an archive will contain a few with a missing slice, two series in one
    folder, or duplicated files, and stopping at the first one means looking at them one at a
    time for an afternoon. The point of this function is the list.

    No pixels are read: every check here runs on headers, so this stays usable on a hundred
    gigabytes of data.
    """
    from pyplatypus.data.dicom_series import describe_series
    from pyplatypus.errors import PlatypusError

    rows = []
    for path in paths:
        row = {
            "path": str(path), "ok": False, "slices": 0, "sorted_by": None,
            "spacing": None, "series_uid": None, "problem": None,
        }
        try:
            series = describe_series(path)
        except PlatypusError as error:
            row["problem"] = str(error)
        except Exception as error:  # noqa: BLE001
            row["problem"] = f"{type(error).__name__}: {error}"
        else:
            row.update({
                "ok": True,
                "slices": len(series),
                "sorted_by": series.sorted_by,
                "spacing": list(series.spacing),
                "series_uid": series.series_uid,
                "shape": list(series.shape),
            })
        rows.append(row)
    return {"ok": True, "series": rows}


def series_support() -> bool:
    try:
        from pyplatypus.data import dicom_series  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def transform_names(rank: int = 2) -> dict:
    """Which albumentations transforms can be used, at this rank.

    For volumes the list has to be found by trying rather than read from anywhere: support is
    uneven and a transform that cannot take a volume raises from inside the library, which is
    why the engine probes. Slow enough to be worth doing once, so it is cached there.
    """
    import pyplatypus

    try:
        from pyplatypus.spec.components import available_transforms
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)

    try:
        names = sorted(available_transforms(rank=int(rank)))
    except pyplatypus.PlatypusError as error:
        return _failure(error)
    except Exception as error:  # noqa: BLE001
        return _engine_failure(error)
    return {"ok": True, "transforms": names, "rank": int(rank)}
