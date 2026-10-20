"""Spawn-safe process workers. Shared distributions are installed once per worker."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from .pipeline import _advance_image, _ImageTask, _ImageUpdate


_shared: tuple | None = None


def _initialize_worker(shared: tuple) -> None:
    global _shared
    _shared = shared


def advance_image_in_worker(task: _ImageTask) -> _ImageUpdate:
    if _shared is None:
        raise RuntimeError("DDC worker was not initialized")
    return _advance_image(task, _shared)


def create_pool(workers: int, shared: tuple) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(shared,),
    )


def completed_images(pool: ProcessPoolExecutor, tasks: list[_ImageTask]):
    """Yield each image promptly, even when an earlier image is still busy."""
    # Start expensive ROIs first so a large late ROI does not leave the other
    # workers idle at the end of a pass. RNG and output storage are per image.
    ordered = sorted(tasks, key=lambda task: len(task.locations), reverse=True)
    futures = {pool.submit(advance_image_in_worker, task): task for task in ordered}
    try:
        for future in as_completed(futures):
            yield futures[future], future.result()
    finally:
        for future in futures:
            future.cancel()
