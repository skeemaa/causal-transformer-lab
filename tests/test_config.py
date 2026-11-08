"""Tests for model architecture configuration."""

import pytest

from causal_transformer_lab.config import ModelConfig, get_model_config


def test_model_config_stores_architecture_settings():
    """Retain the supplied model architecture settings."""

    config = ModelConfig(
        vocab_size=93,
        context_length=32,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )

    assert config.vocab_size == 93
    assert config.context_length == 32
    assert config.embedding_dim == 64
    assert config.num_heads == 4
    assert config.num_layers == 2
    assert config.dropout == 0.0


def test_model_config_rejects_uneven_head_dimensions():
    """Reject embedding dimensions that cannot split evenly across heads."""

    with pytest.raises(ValueError):
        ModelConfig(
            vocab_size=93,
            context_length=32,
            embedding_dim=65,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )


@pytest.mark.parametrize(
    "field",
    [
        "vocab_size",
        "context_length",
        "embedding_dim",
        "num_heads",
        "num_layers",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_model_config_rejects_nonpositive_sizes(field, value):
    """Reject zero or negative architecture dimensions and counts."""

    settings = {
        "vocab_size": 93,
        "context_length": 32,
        "embedding_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.0,
    }
    settings[field] = value

    with pytest.raises(ValueError):
        ModelConfig(**settings)


@pytest.mark.parametrize("dropout", [-0.1, 1.1])
def test_model_config_rejects_invalid_dropout(dropout):
    """Reject dropout probabilities outside zero through one."""

    with pytest.raises(ValueError):
        ModelConfig(
            vocab_size=93,
            context_length=32,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=dropout,
        )


@pytest.mark.parametrize("dropout", [0.0, 0.1, 1.0])
def test_model_config_accepts_valid_dropout(dropout):
    """Accept dropout probabilities including both boundaries."""
    config = ModelConfig(
        vocab_size=93,
        context_length=32,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=dropout,
    )

    assert config.dropout == dropout


@pytest.mark.parametrize(
    ("name", "context_length", "embedding_dim", "num_heads", "num_layers", "dropout"),
    [
        ("micro", 32, 64, 4, 2, 0.0),
        ("tiny", 256, 256, 8, 6, 0.1),
        ("small", 256, 384, 12, 6, 0.1),
    ],
)
def test_get_model_config_matches_design(
    name, context_length, embedding_dim, num_heads, num_layers, dropout
):
    """Construct each documented preset with the supplied vocabulary size."""

    config = get_model_config(name, vocab_size=7)

    assert config == ModelConfig(
        vocab_size=7,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def test_get_model_config_rejects_unknown_name():
    """Reject an unrecognized preset name."""

    with pytest.raises(ValueError):
        get_model_config("unknown", vocab_size=7)
