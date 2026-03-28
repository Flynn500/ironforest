"""Random number generation backed by a fast Rust PRNG.

Example::

    from ironforest._core import random

    rng = random.Generator.from_seed(42)
    samples = rng.normal(mu=0.0, sigma=1.0, shape=[100, 2])
"""

from typing import Sequence
from ironforest._core import Array


class Generator:
    """Random number generator."""

    def __init__(self) -> None:
        """Create time-seeded generator."""
        ...

    @staticmethod
    def from_seed(seed: int) -> Generator:
        """Create generator with explicit seed."""
        ...

    @staticmethod
    def new() -> Generator:
        """Create a generator."""
        ...

    def uniform(self, low: float, high: float, shape: Sequence[int]) -> Array[float]: ...
    def standard_normal(self, shape: Sequence[int]) -> Array[float]: ...
    def normal(self, mu: float, sigma: float, shape: Sequence[int]) -> Array[float]: ...
    def randint(self, low: int, high: int, shape: Sequence[int]) -> Array: ...

    def gamma(self, shape_param: float, scale: float, shape: Sequence[int]) -> Array[float]:
        """Generate gamma-distributed random samples.

        Args:
            shape_param: Shape parameter (k or alpha), must be positive.
            scale: Scale parameter (theta), must be positive.
            shape: Output array shape.

        Returns:
            Array of gamma-distributed samples.
        """
        ...

    def beta(self, alpha: float, beta: float, shape: Sequence[int]) -> Array[float]:
        """Generate beta-distributed random samples.

        Args:
            alpha: First shape parameter, must be positive.
            beta: Second shape parameter, must be positive.
            shape: Output array shape.

        Returns:
            Array of beta-distributed samples in the interval (0, 1).
        """
        ...

    def lognormal(self, mu: float, sigma: float, shape: Sequence[int]) -> Array[float]:
        """Generate log-normal distributed random samples.

        Args:
            mu: Mean of the underlying normal distribution.
            sigma: Standard deviation of the underlying normal distribution.
            shape: Output array shape.

        Returns:
            Array of log-normal distributed samples.
        """
        ...


def seed(seed: int) -> Generator:
    """Create a seeded random number generator."""
    ...
