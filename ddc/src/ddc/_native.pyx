"""C kernels for guarded double-precision DDC inputs.

Compilation disables fast-math and floating-point contraction. Linking uses
NumPy's mean reduction in original column-major pair order. Ambiguous distance
bins return None so the caller reruns the unchanged NumPy implementation.
"""

import numpy as np
from libc.math cimport sqrt, floor, fabs, pow
from libc.float cimport DBL_EPSILON


def distance_bins(const double[:, ::1] locations, const double[::1] frames,
                  Py_ssize_t n_bins, double horizon):
    cdef Py_ssize_t n = frames.shape[0], d = locations.shape[1]
    cdef Py_ssize_t i, j, k, index, position
    cdef double difference, delta, squared
    counts_array = np.zeros(n_bins, dtype=np.intp)
    cdef Py_ssize_t[::1] counts = counts_array
    for i in range(n):
        for j in range(i + 1, n):
            difference = frames[j] - frames[i]
            if difference >= horizon + 1:
                break
            if difference > 0:
                index = <Py_ssize_t>difference - 1
                if index >= n_bins:
                    raise IndexError(f"DistanceMatrix{{{index + 1}}} does not exist")
                counts[index] += 1
    offsets_array = np.zeros(n_bins + 1, dtype=np.intp)
    cdef Py_ssize_t[::1] offsets = offsets_array
    for k in range(n_bins):
        offsets[k + 1] = offsets[k] + counts[k]
    positions_array = offsets_array[:-1].copy()
    cdef Py_ssize_t[::1] positions = positions_array
    values_array = np.empty(offsets[n_bins], dtype=np.float64)
    cdef double[::1] values = values_array
    with nogil:
        for i in range(n):
            for j in range(i + 1, n):
                difference = frames[j] - frames[i]
                if difference >= horizon + 1:
                    break
                if difference > 0:
                    index = <Py_ssize_t>difference - 1
                    squared = 0.0
                    for k in range(d):
                        delta = locations[i, k] - locations[j, k]
                        squared += delta * delta
                    position = positions[index]
                    values[position] = pow(squared, 0.5)
                    positions[index] += 1
    return values_array, offsets_array


cdef Py_ssize_t lower_bound(const double[::1] frames, double value) noexcept nogil:
    cdef Py_ssize_t left = 0, right = frames.shape[0], middle
    while left < right:
        middle = (left + right) // 2
        if frames[middle] < value:
            left = middle + 1
        else:
            right = middle
    return left


def eliminate(const double[:, ::1] locations, const double[::1] frames,
              const double[::1] constants, const double[:, ::1] deviation,
              const double[::1] labels, double resolution, Py_ssize_t horizon,
              bint probability_mode):
    cdef Py_ssize_t n = frames.shape[0], d = locations.shape[1]
    cdef Py_ssize_t difference, source, target, i, j, k, left_count, right_count
    cdef Py_ssize_t left_index, right_index, lag, distance_bin, count, needed
    cdef double left_square, right_square, dot, squared, boundary, error_bound
    cdef double max_lag, frame_lag, probability, source_label, target_label
    linked_array = np.zeros(n, dtype=np.uint8)
    blink_array = np.zeros(n, dtype=np.uint8)
    trajectory_array = np.zeros(n) if probability_mode else np.asarray(labels).copy()
    probability_array = np.zeros(n)
    left_array = np.empty(n, dtype=np.intp)
    right_array = np.empty(n, dtype=np.intp)
    selected_array = np.empty(max(n, 1))
    cdef unsigned char[::1] linked = linked_array, blink = blink_array
    cdef double[::1] trajectory = trajectory_array, probabilities = probability_array
    cdef Py_ssize_t[::1] left_members = left_array, right_members = right_array
    cdef double[::1] selected = selected_array
    cdef bint invalid_lag

    for difference in range(1, horizon + 1):
        for source in range(n):
            target = lower_bound(frames, frames[source] + difference)
            if target >= n:
                break
            while target < n and frames[target] - frames[source] == difference:
                if linked[source] == 0 and blink[target] == 0:
                    source_label = trajectory[source]
                    target_label = trajectory[target]
                    left_count = right_count = 0
                    if source_label == 0:
                        left_members[0] = source
                        left_count = 1
                    else:
                        for i in range(n):
                            if trajectory[i] == source_label:
                                left_members[left_count] = i
                                left_count += 1
                    if target_label == 0:
                        right_members[0] = target
                        right_count = 1
                    else:
                        for i in range(n):
                            if trajectory[i] == target_label:
                                right_members[right_count] = i
                                right_count += 1

                    # With small integer frames the source squared expression
                    # and absolute subtraction yield the same exact lag.
                    max_lag = 0.0
                    invalid_lag = False
                    for j in range(right_count):
                        for i in range(left_count):
                            frame_lag = fabs(frames[left_members[i]] - frames[right_members[j]])
                            if frame_lag > max_lag:
                                max_lag = frame_lag
                            if frame_lag < 1:
                                invalid_lag = True
                    if max_lag < horizon:
                        if invalid_lag:
                            raise IndexError("MATLAB sub2ind received invalid subscripts")
                        needed = left_count * right_count
                        if needed > selected.shape[0]:
                            selected_array = np.empty(needed)
                            selected = selected_array
                        count = 0
                        # MATLAB (:), then sub2ind: target column outermost.
                        for j in range(right_count):
                            right_index = right_members[j]
                            for i in range(left_count):
                                left_index = left_members[i]
                                left_square = right_square = dot = 0.0
                                for k in range(d):
                                    left_square += locations[left_index, k] * locations[left_index, k]
                                    right_square += locations[right_index, k] * locations[right_index, k]
                                    dot += locations[left_index, k] * locations[right_index, k]
                                squared = left_square + right_square - 2.0 * dot
                                error_bound = 64.0 * DBL_EPSILON * (left_square + right_square + 1.0)
                                if squared <= error_bound:
                                    return None
                                distance_bin = <Py_ssize_t>floor(sqrt(squared) / resolution)
                                boundary = distance_bin * resolution
                                if distance_bin > 0 and fabs(squared - boundary * boundary) <= error_bound:
                                    return None
                                boundary = (distance_bin + 1) * resolution
                                if fabs(squared - boundary * boundary) <= error_bound:
                                    return None
                                if distance_bin >= deviation.shape[1]:
                                    distance_bin = deviation.shape[1] - 1
                                lag = <Py_ssize_t>fabs(frames[left_index] - frames[right_index]) - 1
                                selected[count] = deviation[lag, distance_bin]
                                count += 1
                        probability = float(np.mean(selected_array[:count]))
                        if not probability_mode:
                            probability /= fabs(1.0 + constants[target])
                        if probability > 0.5:
                            blink[target] = 1
                            linked[source] = 1
                            if probability_mode:
                                probabilities[target] = probability
                            if trajectory[source] == 0:
                                trajectory[source] = labels[source]
                            if target_label != 0:
                                for i in range(n):
                                    if trajectory[i] == target_label:
                                        trajectory[i] = trajectory[source]
                            else:
                                trajectory[target] = trajectory[source]
                        elif probability_mode and probability > probabilities[target]:
                            probabilities[target] = probability
                target += 1
    return blink_array == 0, probability_array, trajectory_array
