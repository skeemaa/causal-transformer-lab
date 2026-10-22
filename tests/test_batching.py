"""Tests for training windows and batching."""

import pytest
import torch

from causal_transformer_lab.batching import get_window, sample_batch


def test_get_window_shifts_targets_by_one():
    """Extract inputs and targets from the requested start position."""
    tokens = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.long)

    inputs, targets = get_window(tokens, start=1, context_length=3)

    assert torch.equal(inputs, torch.tensor([20, 30, 40], dtype=torch.long))
    assert torch.equal(targets, torch.tensor([30, 40, 50], dtype=torch.long))


def test_get_window_preserves_shape_and_integer_dtype():
    """Return context-length vectors containing integer token IDs."""
    tokens = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)

    inputs, targets = get_window(tokens, start=0, context_length=3)

    assert inputs.shape == (3,)
    assert targets.shape == (3,)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


@pytest.mark.parametrize(
    ("start", "expected_inputs", "expected_targets"),
    [
        (0, [10, 20, 30], [20, 30, 40]),
        (2, [30, 40, 50], [40, 50, 60]),
    ],
)
def test_get_window_accepts_boundary_starts(start, expected_inputs, expected_targets):
    """Accept the earliest and latest valid window starts."""
    tokens = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.long)

    inputs, targets = get_window(tokens, start=start, context_length=3)

    assert torch.equal(inputs, torch.tensor(expected_inputs, dtype=torch.long))
    assert torch.equal(targets, torch.tensor(expected_targets, dtype=torch.long))


@pytest.mark.parametrize(
    ("start", "context_length"),
    [
        (0, 0),  # Empty context
        (0, -1),  # Negative context
        (-1, 3),  # Negative start
        (3, 3),  # Start beyond the latest valid position
        (0, 6),  # Entire sequence is too short for this context
    ],
)
def test_get_window_rejects_invalid_requests(start, context_length):
    """Rejects requests that cannot produce a complete training window."""
    tokens = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.long)

    with pytest.raises(ValueError):
        get_window(tokens, start=start, context_length=context_length)


def test_sample_batch_returns_shifted_integer_rows():
    """Build a batch of complete windows with next-token targets."""
    tokens = torch.arange(8, dtype=torch.long)

    inputs, targets = sample_batch(
        tokens,
        batch_size=2,
        context_length=3,
    )

    assert inputs.shape == (2, 3)
    assert targets.shape == (2, 3)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long

    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert torch.equal(targets, inputs + 1)

    starts = inputs[:, 0]
    assert bool(torch.all((starts >= 0) & (starts <= 4)))


def test_sample_batch_is_reproducible_with_same_seed():
    """Produce identical batches from generators with the same seed."""
    tokens = torch.arange(20, dtype=torch.long)
    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)

    first_inputs, first_targets = sample_batch(
        tokens,
        batch_size=4,
        context_length=3,
        generator=first_generator,
    )
    second_inputs, second_targets = sample_batch(
        tokens,
        batch_size=4,
        context_length=3,
        generator=second_generator,
    )

    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_targets, second_targets)


@pytest.mark.parametrize(
    ("batch_size", "context_length"),
    [
        (0, 3),
        (-1, 3),
        (2, 0),
        (2, -1),
        (2, 6),
    ],
)
def test_sample_batch_rejects_invalid_requests(batch_size, context_length):
    """Reject invalid sizes and insufficient source tokens."""
    tokens = torch.arange(6, dtype=torch.long)

    with pytest.raises(ValueError):
        sample_batch(
            tokens,
            batch_size=batch_size,
            context_length=context_length,
        )


def test_sample_batch_accepts_one_available_window():
    """Fill a batch when only one complete window is available."""
    tokens = torch.tensor([10, 20, 30, 40], dtype=torch.long)

    inputs, targets = sample_batch(
        tokens,
        batch_size=2,
        context_length=3,
    )

    assert torch.equal(
        inputs,
        torch.tensor([[10, 20, 30], [10, 20, 30]], dtype=torch.long),
    )
    assert torch.equal(
        targets,
        torch.tensor([[20, 30, 40], [20, 30, 40]], dtype=torch.long),
    )
