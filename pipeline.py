"""Occupational similarity pipeline.

Runs all steps end-to-end:
  1. Download ISCO-08 data (or load fallback)
  2. Parse the Excel file
  3. Generate sentence embeddings (all-mpnet-base-v2)
  4. Compute cosine similarity matrices
  5. Reduce dimensionality (Kernel PCA + UMAP)
  6. Visualize UMAP projections

Usage:
    python pipeline.py
"""


import numpy as np

from src.download import download_isco
from src.embed import generate_embeddings
from src.parse import parse_isco
from src.reduce import reduce_embeddings
from src.similarity import compute_similarity
from src.visualize import plot_umap
from src.logging_config import configure_logging

configure_logging()

def main() -> None:
    np.random.seed(420)

    # Step 1: Acquire data
    path = download_isco()

    # Step 2: Parse
    df = parse_isco(path)

    # Step 3: Embed
    embeddings = generate_embeddings(df)

    # Step 4: Similarity
    compute_similarity(embeddings, codes=df["code"].tolist())

    # Step 5: Dimensionality reduction
    reduced = reduce_embeddings(embeddings, df)

    # Step 6: Visualize
    fig_path = plot_umap(reduced)
    print(f"\nDone. Figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
