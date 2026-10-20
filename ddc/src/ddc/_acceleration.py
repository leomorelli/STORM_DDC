"""Conservative dispatch to optional compiled kernels."""

import os
import numpy as np

try:
    from . import _native
except ImportError:
    _native = None


def native_available():
    return _native is not None and os.environ.get("DDC_DISABLE_NATIVE") != "1"


def try_eliminate(locations, frames, constants, resolution, a, deviation, labels, *, probability_mode):
    if not native_available():
        return None
    if labels.dtype.kind not in "iuf":
        return None
    labels = np.asarray(labels, dtype=np.float64)
    if (locations.dtype != np.float64 or frames.dtype != np.float64
        or locations.ndim != 2 or locations.shape[1] not in (2, 3)
        or locations.shape[0] != frames.size or frames.size < locations.shape[1]
        or deviation.dtype != np.float64 or deviation.ndim != 2 or deviation.shape[0] < a or not deviation.shape[1]
        or not np.all(np.isfinite(locations)) or not np.all(np.abs(locations) < 1e8)
        or not np.all(np.isfinite(frames)) or not np.all(frames == np.floor(frames))
        or not np.all(np.abs(frames) < 2**25) or not np.all(frames[1:] >= frames[:-1])
        or not np.all(np.isfinite(labels)) or labels.size != frames.size
        or not np.isfinite(resolution) or not 1e-6 <= resolution <= 1e12):
        return None
    if not probability_mode and (constants.dtype != np.float64 or constants.ndim != 1
        or constants.size != frames.size
        or not np.all(np.isfinite(constants)) or np.any(constants == -1)):
        return None
    return _native.eliminate(
        np.ascontiguousarray(locations), np.ascontiguousarray(frames),
        np.ascontiguousarray(constants, dtype=float), np.ascontiguousarray(deviation, dtype=float),
        np.ascontiguousarray(labels, dtype=float), float(resolution), int(a), probability_mode,
    )


def try_distance_bins(locations, frames, cells, horizon):
    if not native_available():
        return None
    if (locations.dtype != np.float64 or frames.dtype != np.float64
        or locations.shape[1] not in (2, 3)
        or not np.all(np.isfinite(locations)) or not np.all(np.abs(locations) < 1e150)
        or not np.all(np.isfinite(frames)) or not np.all(frames == np.floor(frames))
        or not np.all(np.abs(frames) < 2**25)
        or any(cell.dtype != np.float64 for cell in cells)):
        return None
    values, offsets = _native.distance_bins(np.ascontiguousarray(locations),
        np.ascontiguousarray(frames), len(cells), float(horizon))
    return [np.concatenate((cell, values[offsets[i]:offsets[i + 1]]))
            if offsets[i] != offsets[i + 1] else cell
            for i, cell in enumerate(cells)]
