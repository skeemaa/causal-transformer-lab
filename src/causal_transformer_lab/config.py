"""Define model architecture configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Stores the settings needed to construct the model"""

    vocab_size: int
    context_length: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    dropout: float

    def __post_init__(self) -> None:
        """Validate architecture constraints after initialization."""

        sizes = (
            self.vocab_size,
            self.context_length,
            self.embedding_dim,
            self.num_heads,
            self.num_layers,
        )

        # Validate all settings are positive
        if any(size <= 0 for size in sizes):
            raise ValueError("Model dimensions and counts must be positive.")

        # Validates embedding dim
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by head count.")

        # Validate dropout
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("Dropout must be between zero and one.")


def get_model_config(name: str, *, vocab_size: int) -> ModelConfig:
    """Construct a named architecture preset for the supplied vocabulary."""

    if name == "micro":
        return ModelConfig(
            vocab_size=vocab_size,
            context_length=32,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )
    if name == "tiny":
        return ModelConfig(
            vocab_size=vocab_size,
            context_length=256,
            embedding_dim=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1,
        )
    if name == "small":
        return ModelConfig(
            vocab_size=vocab_size,
            context_length=256,
            embedding_dim=384,
            num_heads=12,
            num_layers=6,
            dropout=0.1,
        )

    raise ValueError(f"Unknown model preset: {name}")
