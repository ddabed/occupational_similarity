import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def compute_similarity(
    embeddings: dict[str, np.ndarray],
    codes: list[str],
    output_dir: str = "outputs/similarity",
) -> dict[str, np.ndarray]:
    """Compute pairwise cosine similarity matrices for each embedding field.

    Args:
        embeddings: Dict mapping field name → embedding array of shape (N, D).
        codes: Occupation codes used as row/column labels in the output CSV.
        output_dir: Directory to save .csv similarity matrix files.

    Returns:
        Dict mapping field name → cosine similarity matrix of shape (N, N).
    """
    os.makedirs(output_dir, exist_ok=True)
    similarity_matrices: dict[str, np.ndarray] = {}

    for field, vecs in embeddings.items():
        logger.info("Computing cosine similarity for field '%s' (shape %s)...", field, vecs.shape)
        sim = cosine_similarity(vecs)
        out_path = os.path.join(output_dir, f"{field}.csv")
        pd.DataFrame(sim, index=codes, columns=codes).to_csv(out_path, index=True)
        logger.info("Saved similarity matrix to %s (shape %s)", out_path, sim.shape)
        similarity_matrices[field] = sim

    return similarity_matrices
