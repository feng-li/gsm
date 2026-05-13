"""Configuration objects for the FEBAMA application layer."""

from dataclasses import dataclass

from gsm.febama.distributions import get_distribution


@dataclass(frozen=True)
class FebamaConfig:
    """Minimal data-source-agnostic FEBAMA configuration."""

    model_names: tuple[str, ...]
    distribution_names: tuple[str, ...]
    add_intercept: bool = True

    def __post_init__(self) -> None:
        if len(self.model_names) < 2:
            raise ValueError("FEBAMA requires at least two component models")
        if len(self.distribution_names) != len(self.model_names):
            raise ValueError("distribution_names must match model_names")
        for name in self.distribution_names:
            get_distribution(name)
