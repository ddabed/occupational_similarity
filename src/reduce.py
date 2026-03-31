import logging
import os

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import KernelPCA

logger = logging.getLogger(__name__)


def reduce_embeddings(
    embeddings: dict[str, np.ndarray],
    df_meta: pd.DataFrame,
    output_dir: str = "outputs/reduced",
    random_state: int = 420,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Reduce embeddings to 2D using Kernel PCA and UMAP.

    Args:
        embeddings: Dict mapping field name → embedding array of shape (N, D).
        df_meta: DataFrame with occupation metadata (code, title, major_group).
        output_dir: Directory to save 2D coordinate CSV files.
        random_state: Random seed for reproducibility.

    Returns:
        Nested dict: {field: {"kpca": DataFrame, "umap": DataFrame}}
        Each DataFrame has columns: x, y, code, title, major_group.
    """
    os.makedirs(output_dir, exist_ok=True)

    meta_cols = ["code", "title", "major_group"]
    meta = df_meta[meta_cols].reset_index(drop=True)

    results: dict[str, dict[str, pd.DataFrame]] = {}

    for field, vecs in embeddings.items():
        results[field] = {}

        # --- Kernel PCA ---
        logger.info("Running Kernel PCA for field '%s'...", field)
        kpca = KernelPCA(n_components=2, kernel="rbf", random_state=random_state)
        coords_kpca = kpca.fit_transform(vecs)
        df_kpca = _make_coord_df(coords_kpca, meta)
        kpca_path = os.path.join(output_dir, f"{field}_kpca.csv")
        df_kpca.to_csv(kpca_path, index=False)
        logger.info("Saved Kernel PCA coordinates to %s", kpca_path)
        results[field]["kpca"] = df_kpca

        # --- UMAP ---
        logger.info("Running UMAP for field '%s'...", field)
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=random_state)
        coords_umap = reducer.fit_transform(vecs)
        df_umap = _make_coord_df(coords_umap, meta)
        umap_path = os.path.join(output_dir, f"{field}_umap.csv")
        df_umap.to_csv(umap_path, index=False)
        logger.info("Saved UMAP coordinates to %s", umap_path)
        results[field]["umap"] = df_umap

    return results


def _make_coord_df(coords: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    return pd.concat([df, meta.reset_index(drop=True)], axis=1)
