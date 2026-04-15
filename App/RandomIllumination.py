"""Manager for multiple animated light sources.

Draws multiple animated light sources that move smoothly across the screen
using cubic spline interpolation.
"""

from typing import Any, List

from App.IlluminationSource import IlluminationSource


class RandomIllumination:
    """Manager for multiple animated light sources.

    Coordinates multiple illumination sources and updates/renders them
    each frame for dynamic lighting effects.

    Attributes:
        _sources: List of IlluminationSource instances
    """

    def __init__(self, sources_n: int = 32) -> None:
        """Initialize illumination manager with multiple sources.

        Args:
            sources_n: Number of light sources to create (default: 32).

        Raises:
            ValueError: If sources_n is not positive.
        """
        if sources_n <= 0:
            raise ValueError(f"sources_n must be positive, got: {sources_n}")
        self._sources: List[IlluminationSource] = [
            IlluminationSource() for _ in range(sources_n)
        ]

    def on_tick(self, delta_t: float) -> None:
        """Update all illumination sources.

        Args:
            delta_t: Time delta since last frame in seconds.
        """
        for source in self._sources:
            source.on_tick(delta_t=delta_t)

    def on_render(self, window: Any) -> None:
        """Render all illumination sources.

        Args:
            window: Pygame surface to render to.
        """
        for source in self._sources:
            source.on_render(window=window)
