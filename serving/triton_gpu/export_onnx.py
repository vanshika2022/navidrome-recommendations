"""
Export the recommendation dot product as an ONNX model for Triton.
Adapted from lab's torch.onnx.export pattern.

This wraps the dot product in a simple PyTorch module so we can
export it to ONNX format, which Triton can serve.
"""

import torch
import torch.nn as nn
import numpy as np

# Same dummy embeddings as other serving options
NUM_SONGS = 10000
EMBEDDING_DIM = 64

np.random.seed(42)
# We don't need user embeddings in the model — just song embeddings
# The user embedding is passed as input at inference time
song_embeddings_np = np.random.randn(NUM_SONGS, EMBEDDING_DIM).astype(np.float32)


class RecommenderModel(nn.Module):
    """Wraps the dot product in a PyTorch module so we can export to ONNX.
    Input: user embedding vector (1 x 64)
    Output: scores for all songs (1 x 10000)
    """
    def __init__(self, song_embeddings):
        super().__init__()
        # Store song embeddings as a fixed parameter (not trainable)
        self.song_embeddings = nn.Parameter(
            torch.tensor(song_embeddings), requires_grad=False
        )

    def forward(self, user_embedding):
        # Same dot product as baseline: user_emb @ song_emb.T
        scores = torch.matmul(user_embedding, self.song_embeddings.T)
        return scores


# Create model and export to ONNX (same pattern as lab's torch.onnx.export)
model = RecommenderModel(song_embeddings_np)
model.eval()

# Dummy input for export (1 user embedding of 64 dims)
dummy_input = torch.randn(1, EMBEDDING_DIM)

# Export (same as lab: torch.onnx.export(model, dummy_input, path, ...))
onnx_path = "model_repository/song_recommender/1/model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    input_names=["user_embedding"],
    output_names=["scores"],
    dynamic_axes={
        "user_embedding": {0: "batch_size"},
        "scores": {0: "batch_size"},
    },
)

print(f"ONNX model exported to {onnx_path}")
