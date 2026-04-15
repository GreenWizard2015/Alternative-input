"""Sampling strategies for frame selection in temporal trajectories.

Provides different strategies for selecting frames from temporal sequences,
including uniform sampling, time-based sampling, and advanced sampling with
configurable parameters.
"""

from typing import List, Any, Dict
import random
import numpy as np


def uniform_time_sampling(
    samples: List[int],
    steps: int,
    mainInd: int,
    storage: Any,
    **kwargs: Any,  # type: ignore[unused-argument]
) -> List[int]:
    """Sample frames uniformly distributed across time window.

    Selects 'steps' frames from candidates with uniform time intervals.
    Ensures frames are strictly ordered by time.

    Unused kwargs are kept for API compatibility with unified sampling interface.

    Args:
        samples: Available frame candidates
        steps: Number of frames to select
        mainInd: Index of main/reference frame
        storage: Storage object to access frame times
        **kwargs: Additional parameters from unified interface (kept for API compatibility)

    Returns:
        List of selected frame indices (length == steps - 1)
    """
    samples = list(sorted(samples))
    samples.append(mainInd)
    minInd, maxInd = samples[0], samples[-1]
    minT = storage[minInd]["time"]
    T = storage[maxInd]["time"] - minT
    if not (0 < T):
        raise ValueError("Time difference is zero")

    times_list = [storage[ind]["time"] - minT for ind in samples]
    times: np.ndarray = np.array(times_list)
    if not np.all(0 < np.diff(times)):
        raise ValueError("Time is not strictly increasing")
    if not (steps <= len(samples)):
        raise ValueError("Not enough samples to sample frames")

    candidates = []
    t_list = np.linspace(0.0, T, num=steps)
    assert len(t_list) == steps, f"Expected {steps} samples, got {len(t_list)}"
    for t in t_list:
        # find the frame with time nearest to t
        idx = np.argmin(np.abs(times - t))
        candidates.append(samples[idx])
        # remove the frame from the list and time
        samples.pop(idx)
        times = np.delete(times, idx)

    result = list(sorted(candidates))
    return result[:-1]  # remove mainIdx


def uniform_sampling(
    samples: List[int], steps: int, **kwargs: Any  # type: ignore[unused-argument]
) -> List[int]:
    """Randomly sample frames uniformly from candidates.

    Unused kwargs are kept for API compatibility with unified sampling interface.

    Args:
        samples: Available frame candidates
        steps: Number of frames to select
        **kwargs: Additional parameters from unified interface (kept for API compatibility)

    Returns:
        List of randomly selected frame indices (length == steps - 1)
    """
    return random.sample(population=samples, k=steps - 1)


def last_sampling(
    samples: List[int], steps: int, **kwargs: Any  # type: ignore[unused-argument]
) -> List[int]:
    """Select the most recent frames from candidates.

    Unused kwargs are kept for API compatibility with unified sampling interface.

    Args:
        samples: Available frame candidates (ordered by index)
        steps: Number of frames to select
        **kwargs: Additional parameters from unified interface (kept for API compatibility)

    Returns:
        List of most recent frame indices (length == steps - 1)
    """
    return samples[-(steps - 1) :]


def dict_sampling(
    samples: List[int],
    steps: int,
    sampling: Dict[str, Any],
    **kwargs: Any,  # type: ignore[unused-argument]
) -> List[int]:
    """Advanced sampling strategy with configurable max frames.

    Unused kwargs are kept for API compatibility with unified sampling interface.

    Args:
        samples: Available frame candidates
        steps: Number of frames to select
        sampling: Dictionary with 'max frames' parameter
        **kwargs: Additional parameters from unified interface (kept for API compatibility)

    Returns:
        List of selected frame indices (length == steps - 1)
    """
    candidates = list(samples)
    maxFrames: int = sampling["max frames"]
    candidates = candidates[::-1]
    result: List[int] = []
    left = steps - 1
    for _ in range(left):
        avl = min((maxFrames, 1 + len(candidates) - left))
        ind = random.randint(0, avl - 1)
        result.append(candidates[ind])
        candidates = candidates[ind + 1 :]
        left -= 1
    return result
