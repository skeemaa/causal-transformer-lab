"""Combine learned token and absolute positional embeddings."""

import torch
from torch import nn

from causal_transformer_lab.config import ModelConfig


class TokenPositionEmbedding(nn.Module):
    """Represent token identity and position with learned vectors."""

    def __init__(self, config: ModelConfig) -> None:
        """Create embedding tables and embedding dropout."""

        super().__init__()

        # Table (rows, features)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Table (rows, features)
        self.position_embedding = nn.Embedding(
            config.context_length, config.embedding_dim
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs shaped (B, T) to representations shaped (B, T, C)."""

        sequence_length = token_ids.shape[1]

        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError("Sequence length exceeds the model context length.")

        # Create position IDs
        positions = torch.arange(sequence_length, device=token_ids.device)

        # Lookup token vectors
        token_vectors = self.token_embedding(token_ids)

        # Lookup position vectors
        position_vectors = self.position_embedding(positions)

        # Add together and apply dropout
        return self.dropout(token_vectors + position_vectors)
