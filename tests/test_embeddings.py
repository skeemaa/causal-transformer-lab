"""Tests for token and positional embeddings."""

import pytest
import torch

from causal_transformer_lab.config import ModelConfig
from causal_transformer_lab.embeddings import TokenPositionEmbedding


def test_embeddings_return_batch_sequence_feature_shape():
    """Produce one embedding vector per input token."""

    config = ModelConfig(
        vocab_size=5,
        context_length=4,
        embedding_dim=6,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )

    embeddings = TokenPositionEmbedding(config)
    token_ids = torch.tensor([[1, 0, 1], [2, 3, 4]], dtype=torch.long)

    output = embeddings(token_ids)

    assert output.shape == (2, 3, 6)


def test_embeddings_add_positions_to_selected_token_vectors():
    """Add the same position vectors across batch rows."""

    config = ModelConfig(
        vocab_size=3,
        context_length=3,
        embedding_dim=2,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
    )

    embeddings = TokenPositionEmbedding(config)

    with torch.no_grad():
        embeddings.token_embedding.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
        embeddings.position_embedding.weight.copy_(
            torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        )

    token_ids = torch.tensor([[1, 0, 1], [2, 1, 0]], dtype=torch.long)

    output = embeddings(token_ids)

    expected = torch.tensor(
        [
            [[13.0, 24.0], [31.0, 42.0], [53.0, 64.0]],
            [[15.0, 26.0], [33.0, 44.0], [51.0, 62.0]],
        ]
    )

    torch.testing.assert_close(output, expected)


def test_embeddings_reject_sequences_beyond_context_length():
    """Reject sequences longer than the learned position table."""

    config = ModelConfig(
        vocab_size=5,
        context_length=4,
        embedding_dim=6,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    embeddings = TokenPositionEmbedding(config)
    token_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)

    with pytest.raises(ValueError):
        embeddings(token_ids)


def test_embeddings_apply_dropout_only_during_training():
    """Drop combined representations during training and preserve them in eval."""

    config = ModelConfig(
        vocab_size=3,
        context_length=2,
        embedding_dim=2,
        num_heads=1,
        num_layers=1,
        dropout=1.0,
    )
    embeddings = TokenPositionEmbedding(config)

    with torch.no_grad():
        # Set every table element to the supplied value
        embeddings.token_embedding.weight.fill_(2.0)
        embeddings.position_embedding.weight.fill_(3.0)

    token_ids = torch.tensor([[0, 1]], dtype=torch.long)

    # Includes dropout
    embeddings.train()
    training_output = embeddings(token_ids)

    # Disables dropout
    embeddings.eval()
    evaluation_output = embeddings(token_ids)

    torch.testing.assert_close(training_output, torch.zeros(1, 2, 2))
    torch.testing.assert_close(evaluation_output, torch.full((1, 2, 2), 5.0))


def test_embeddings_propagate_gradients_to_used_table_rows():
    """Accumulate gradients for selected token and position rows."""

    config = ModelConfig(
        vocab_size=3,
        context_length=4,
        embedding_dim=2,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
    )
    embeddings = TokenPositionEmbedding(config)
    token_ids = torch.tensor([[1, 0, 1]], dtype=torch.long)

    output = embeddings(token_ids)
    output.sum().backward()

    expected_token_gradients = torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
    expected_position_gradients = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    )

    torch.testing.assert_close(
        embeddings.token_embedding.weight.grad,
        expected_token_gradients,
    )
    torch.testing.assert_close(
        embeddings.position_embedding.weight.grad,
        expected_position_gradients,
    )
