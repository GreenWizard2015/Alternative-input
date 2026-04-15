"""Interface protocol for data samplers."""

from typing import Any, Protocol, List, Tuple


class SamplerInterface(Protocol):
    """Common interface for data samplers.

    This protocol defines the contract that all sampler implementations must follow.
    Implementers must provide sample management and batch retrieval capabilities.
    """

    @property
    def totalSamples(self) -> int:
        """Get total number of samples available in the sampler.

        Returns:
            Total count of samples stored in this sampler instance.
        """
        ...

    def addBlock(self, data: Any) -> None:
        """Add a data block to the sampler.

        Adds either a list of sample dictionaries or a batched dictionary with
        array values to the underlying storage.

        Args:
            data: Data block to add - either:
                - List of sample dictionaries: [{'time': 0.1, ...}, ...]
                - Batched dict with array values: {'time': [0.1, 0.2], ...}

        Raises:
            ValueError: If data format is invalid.
        """
        ...

    def sampleByIds(
        self, ids: List[int], **kwargs: Any
    ) -> Tuple[Any, List[int], List[int]]:
        """Sample sequences for specific frame indices.

        Attempts to create samples from the provided frame indices, returning
        both accepted samples and rejected indices that failed sampling.

        Args:
            ids: List of frame indices to sample from.
            **kwargs: Additional sampling parameters (timesteps, augmentation settings, etc.).

        Returns:
            Tuple of (result, rejected_ids, accepted_ids) where:
            - result: (X, Y) data tuple or None if no samples produced
            - rejected_ids: Indices that failed sampling
            - accepted_ids: Indices that succeeded

        Raises:
            ValueError: If ids is empty or contains invalid indices.
        """
        ...
