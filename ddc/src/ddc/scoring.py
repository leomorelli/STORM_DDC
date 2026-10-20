"""Numerical PALMplot core used by DDC scoring, without plotting dependencies.

The active source does not match inferred and true points or compute a
precision/recall statistic. ``PALMplot.m`` rasterizes 2-D Gaussian spots in an
image whose bounds come from the truth/reference coordinates. ``DDC_MCMC``
then applies ``mat2gray`` independently to both images and sums squared pixel
differences. The third coordinate column, when present, is localization sigma
in nm, not a z coordinate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matlab_round(values: ArrayLike) -> NDArray[np.float64]:
    """MATLAB default round: nearest integer, ties away from zero."""

    array = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(array) + 0.5), array)


def palm_density_image(coords: ArrayLike, reference_coords: ArrayLike) -> NDArray[np.float64]:
    """Port PALMplot lines 12--105 and return ``Density_Image``.

    No figure, axes, color, file output, or display-only statement is ported.
    """

    points = np.asarray(coords, dtype=np.float64)
    reference = np.asarray(reference_coords, dtype=np.float64)
    if points.ndim != 2 or reference.ndim != 2:
        raise ValueError("coords and reference_coords must be 2D matrices")
    if points.shape[1] < 2 or reference.shape[1] < 2:
        raise ValueError("PALMplot requires at least x and y columns")
    if reference.shape[0] == 0:
        raise ValueError("MATLAB PALMplot cannot determine bounds from empty coords2")

    if points.shape[1] < 3:
        points = np.column_stack((points, np.full(points.shape[0], 40.0)))
        reference = np.column_stack(
            (reference[:, :2], np.full(reference.shape[0], 40.0))
        )

    pixel_size = 1.0
    palm_pixel_size = 10.0
    gaussian_width = 21
    radius = (gaussian_width - 1) // 2
    pixel_multiple = pixel_size / palm_pixel_size

    row_low = np.floor(np.min(reference[:, 1]))
    row_high = np.ceil(np.max(reference[:, 1]))
    column_low = np.floor(np.min(reference[:, 0]))
    column_high = np.ceil(np.max(reference[:, 0]))
    n_rows = row_high - row_low + 1
    n_columns = column_high - column_low + 1
    zoom_rows = int(_matlab_round(n_rows * pixel_multiple)) + 4 * radius
    zoom_columns = int(_matlab_round(n_columns * pixel_multiple)) + 4 * radius
    if zoom_rows < 0 or zoom_columns < 0:
        raise ValueError("MATLAB zeros received a negative PALM image dimension")

    grid_values = np.arange(1, gaussian_width + 1, dtype=np.float64)
    x_palm, y_palm = np.meshgrid(grid_values, grid_values)
    gaussian_center = (gaussian_width + 1) / 2

    dev_x = points[:, 2]
    dev_y = points[:, 2]
    x = points[:, 0] * pixel_multiple
    y = points[:, 1] * pixel_multiple
    reference_x = reference[:, 0] * pixel_multiple
    reference_y = reference[:, 1] * pixel_multiple
    x = x - np.min(reference_x)
    y = y - np.min(reference_y)

    image = np.zeros((zoom_rows, zoom_columns), dtype=np.float64)
    for molecule in range(x.size):
        if dev_x[molecule] <= np.inf and dev_y[molecule] <= np.inf:
            y_position = int(_matlab_round(y[molecule])) + 2 * radius
            x_position = int(_matlab_round(x[molecule])) + 2 * radius
            sigma_x = dev_x[molecule] / palm_pixel_size
            sigma_y = dev_y[molecule] / palm_pixel_size
            if not (
                y_position - radius < 1
                or y_position + radius > zoom_rows
                or x_position - radius < 1
                or x_position + radius > zoom_columns
            ):
                y_center = y[molecule] + 2 * radius - y_position
                x_center = x[molecule] + 2 * radius - x_position
                numerator_x = -(x_palm - (gaussian_center + x_center)) ** 2
                numerator_y = -(y_palm - (gaussian_center + y_center)) ** 2
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    spot = 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(
                        numerator_x / (2 * sigma_x**2)
                        + numerator_y / (2 * sigma_y**2)
                    )
                row_start = y_position - radius - 1
                row_stop = y_position + radius
                column_start = x_position - radius - 1
                column_stop = x_position + radius
                image[row_start:row_stop, column_start:column_stop] += spot
    return image


def matlab_mat2gray(image: ArrayLike) -> NDArray[np.float64]:
    """Numerical ``mat2gray(A)`` used by DDC_MCMC."""

    array = np.asarray(image, dtype=np.float64)
    if array.size == 0:
        return array.copy()
    minimum = np.min(array)
    maximum = np.max(array)
    if maximum == minimum:
        return np.zeros(array.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (array - minimum) / (maximum - minimum)
    result[array <= minimum] = 0
    result[array >= maximum] = 1
    return result


def palm_image_squared_error(
    inferred_coords: ArrayLike, true_coords: ArrayLike
) -> float:
    """DDC_MCMC's raw ``scorey11`` before division by ``max_scorey``."""

    inferred_image = palm_density_image(inferred_coords, true_coords)
    true_image = palm_density_image(true_coords, true_coords)
    difference = matlab_mat2gray(inferred_image) - matlab_mat2gray(true_image)
    return float(np.sum(np.abs(difference) ** 2))
