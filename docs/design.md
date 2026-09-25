# Project Design

## Purpose
This is an educational implementation of a decoder-only causal transformer. Here, "causal" means that attention is masked so each token can attend only to itself and earlier tokens; it does not refer to causal inference. The learning task is to deeply understand character-level next-token prediction by building transformer mechanics explicitly with PyTorch tensor operations. The repository also demonstrates testing, documentation, packaging, and reproducibility practices.

## Version 0.1.0 Scope
V0.1.0 will contain the following:

- Corpus preparation and a character tokenizer
- Training-sequence and mini-batch construction
- Decoder-only transformer components
- Training and validation loops
- Autoregressive generation with greedy, temperature, and top-k sampling
- Checkpoint save/restore
- `micro`, `tiny`, and optional `small` configurations
- Unit/integration tests
- Documentation, packaging, and a reproducible release

## Model Architecture

The following symbols describe the model's tensor shapes:

- B: batch size
- T: sequence length
- C: embedding dimension
- V: vocabulary size

The model receives token IDs with shape (B,T). Token embeddings are added to learned absolute positional embeddings, and dropout is applied to produce a tensor with shape (B,T,C). The decoder stack transforms the representations while preserving this shape. Causal attention prevents each position from attending to future positions.

### Signal Flow:

- Token IDs with shape (B,T)
- Token embeddings plus learned absolute positional embeddings
- Embedding dropout
- Repeated pre-normalized decoder blocks:
  - Layer normalization → causal multi-head self-attention → dropout → residual addition
  - Layer normalization → feed-forward network with GELU → dropout → residual addition
  - The feed-forward network expands and contracts the feature dimension: C → 4C → C
- Final layer normalization
- Separate, untied linear vocabulary projection
- Logits with shape (B,T,V)

## Planned Model Configurations

| Configuration | Context | Embedding | Heads | Head dimension | Blocks | Feed-forward | Dropout | Approximate parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `micro` | 32 | 64 | 4 | 16 | 2 | 256 | 0.0 | ~0.1M |
| `tiny` | 256 | 256 | 8 | 32 | 6 | 1024 | 0.1 | ~4.8M |
| `small` | 256 | 384 | 12 | 32 | 6 | 1536 | 0.1 | ~10.8M |

The configurations have the following purposes:

- `micro` supports fast tests, debugging, and tiny-batch overfitting.
- `tiny` is the primary configuration for documented GPU training.
- `small` supports an optional scaling experiment after `tiny` is validated.

Each embedding dimension must equal the number of attention heads multiplied by the head dimension. Named presets will be selected before model construction; switching configurations creates a new model rather than modifying an existing model.

Model architecture settings will remain separate from training settings such as learning rate, batch size, training steps, evaluation interval, and random seed. Checkpoints will record the exact model configuration required to reconstruct the saved model.

## Implementation Boundary

The transformer architecture and its data flow are assembled explicitly from lower-level PyTorch components. We do not reimplementing tensor operations, automatic differentiation, or numerical kernels.

### Permitted PyTorch Building Blocks

The core implementation may use:

- PyTorch tensors and tensor operations
- PyTorch automatic differentiation
- `nn.Module` and `nn.Parameter`
- `nn.Linear`
- `nn.Embedding`
- `nn.LayerNorm`
- Standard activation functions and dropout
- `torch.optim.AdamW`
- PyTorch's numerically stable cross-entropy implementation
- Tensor serialization utilities

### Components Implemented in This Project

The project will implement:

- Character tokenization
- Training-sequence and mini-batch construction
- Query, key, and value projections within the attention module
- Scaled dot-product attention
- Causal masking
- Attention softmax and weighted-value aggregation
- Attention-head splitting and recombination
- Multi-head self-attention orchestration
- Residual paths
- Feed-forward and decoder-block structure
- Decoder-stack orchestration
- Training and evaluation loops
- Autoregressive generation
- Greedy decoding, temperature, and top-k sampling
- Parameter counting
- Checkpoint handling

### Excluded High-Level Implementations

The core model will not use:

- `torch.nn.Transformer`
- `torch.nn.TransformerDecoder`
- `torch.nn.MultiheadAttention`
- PyTorch's fused scaled-dot-product attention
- Hugging Face model classes
- Pretrained transformers
- Copied end-to-end transformer implementations

The project will not implement a custom automatic-differentiation engine.

## Dataset and Learning Task

The model will perform character-level next-token prediction. For a tokenized window containing (T+1) characters, the first (T) tokens form the input and the final (T) tokens shifted by one position form the target. Batched inputs and targets have shape (B,T), and the model produces logits with shape (B,T,V).

The corpus will use selected Project Gutenberg editions of these public-domain works by Arthur Conan Doyle:

- *The Adventures of Sherlock Holmes*
- *A Study in Scarlet*
- *The Hound of the Baskervilles*

Corpus preparation will record:

- Source landing pages
- Retrieval dates
- Public-domain statements
- SHA-256 hashes of source and processed files
- Cleaning and normalization rules
- Character vocabulary
- Work concatenation order
- Deterministic training and validation split procedure

The cleaned corpus will be committed to the repository for reproducibility but excluded from the installable Python wheel. Exact source editions, cleaning rules, and split behavior will be selected and documented during the corpus-preparation milestone. Source or preprocessing changes must update the corresponding hashes and documentation.

## Testing Strategy

Testing is a central part of the implementation rather than a final validation step. Each major component will be introduced with tests and examples before it is integrated into the next level of the model.

Planned coverage includes:

### Tokenization and Data

- Tokenizer encode/decode round trips
- Vocabulary construction
- Invalid-input behavior
- Training-sequence construction
- Mini-batch shapes and target shifting

### Attention

- Known-value attention calculations
- Score scaling
- Causal-mask structure
- Proof that future tokens cannot affect earlier outputs
- Attention probabilities summing to one
- Attention-head splitting and recombination

### Model Components

- Residual connections
- Layer-normalization behavior
- Feed-forward and decoder-block shapes
- Full forward-pass shapes
- Finite losses and gradients
- Parameter counting

### Training, Generation, and Persistence

- Tiny-batch overfitting
- Deterministic greedy generation
- Temperature and top-k behavior
- Checkpoint round trips
- Equivalent predictions after checkpoint restoration

Each major milestone will also include an educational inspection step. This may include decoded examples, selected tensor values, shape traces, causal-mask diagrams, attention heatmaps, probability plots, or training-loss curves. These visual checks support understanding but do not replace automated assertions.

Continuous integration will run only fast, deterministic, CPU-compatible checks. Longer GPU training runs and scaling experiments will remain outside CI and will be documented separately.

## Reproducibility Requirements

Reproducibility means preserving the information needed to reconstruct an experiment and explain its results. The project will record or control:

- Model and training configurations
- Python, PyTorch, NumPy, and relevant dependency versions
- Random seeds for Python, NumPy, and PyTorch
- Dataset sources, hashes, cleaning rules, vocabulary, and split procedure
- Device type and relevant hardware information
- Training commands and important run settings
- Parameter counts and evaluation results

Checkpoints will contain enough information to reconstruct the model and resume training, including the model state, optimizer state, completed training step, model configuration, training configuration, and tokenizer metadata.

Exact bit-for-bit agreement across different operating systems, PyTorch versions, or CPU and GPU hardware is not guaranteed. Where PyTorch provides deterministic behavior, the project will enable or request it when practical and document any remaining limitations.

## Non-Goals

Version 0.1.0 will not attempt to provide:

- A production-scale or state-of-the-art language model
- A pretrained model comparable to modern commercial LLMs
- Distributed or multi-GPU training
- Custom automatic differentiation or CUDA kernels
- Fused or memory-optimized attention implementations
- Subword tokenization
- Long-context optimization
- Exhaustive hyperparameter tuning
- A production inference service or graphical interface
- Quantization, model export, or edge-device deployment

These may be considered as future extensions.
