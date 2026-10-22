"""Construct next-token training windows and batches."""

import torch


def get_window(
    tokens: torch.Tensor,
    start: int,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return input tokens and their next-token targets."""

    if context_length <= 0:
        raise ValueError("Context length cannot be 0 or negative.")
    if start < 0:
        raise ValueError("Start index cannot be negative.")
    if start + context_length + 1 > len(tokens):
        raise ValueError("Stop index cannot exceed sequence length.")

    window = tokens[start : start + context_length + 1]

    inputs = window[:-1]  # Everything except last token
    targets = window[1:]  # Everything after first token

    return inputs, targets


def sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input windows and their next-token targets."""

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")
    if context_length <= 0:
        raise ValueError("Context length must be greater than zero.")
    if len(tokens) < context_length + 1:
        raise ValueError("Sequence is too short for the requested context length.")

    # Generate random starting indices
    starts = torch.randint(
        low=0,
        high=len(tokens) - context_length,
        size=(batch_size,),
        generator=generator,
    )

    input_rows = []
    target_rows = []

    # For each randomly generated start index
    # extract input and target sequences
    for start in starts.tolist():
        inputs, targets = get_window(tokens, start, context_length)
        input_rows.append(inputs)
        target_rows.append(targets)

    return torch.stack(input_rows), torch.stack(target_rows)  # (B, T), (B, T)
