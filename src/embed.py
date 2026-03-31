import contextlib
import logging
import os
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-mpnet-base-v2"
FIELDS = ["title", "definition", "tasks", "included"]

_BOILERPLATE = re.compile(r"^tasks\s+include[:\s]*", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^\s*[\-\*\d]+[\.\)]\s*", re.MULTILINE)
_EXTRA_WHITESPACE = re.compile(r"\s{2,}")


def clean_tasks(text: str) -> str:
    """Strip boilerplate, list markers, and formatting noise from the tasks field."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _BOILERPLATE.sub("", text.strip())
    text = _LIST_MARKER.sub(" ", text)
    text = _EXTRA_WHITESPACE.sub(" ", text)
    return text.strip()


def generate_embeddings(
    df: pd.DataFrame,
    fields: list[str] = FIELDS,
    model_name: str = MODEL_NAME,
    output_dir: str = "outputs/embeddings",
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Generate sentence embeddings for each text field.

    Args:
        df: DataFrame with occupation data (columns: title, definition, tasks, included).
        fields: List of column names to embed.
        model_name: Sentence transformer model name.
        output_dir: Directory to save .csv embedding files.
        batch_size: Encoding batch size.

    Returns:
        Dict mapping field name → numpy array of shape (N, 768).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Loading model: %s", model_name)
    with open(os.devnull, "w") as _devnull, \
         contextlib.redirect_stdout(_devnull), \
         contextlib.redirect_stderr(_devnull):
        model = SentenceTransformer(model_name)

    embeddings: dict[str, np.ndarray] = {}

    for field in fields:
        if field not in df.columns:
            logger.warning("Field '%s' not found in DataFrame; skipping", field)
            continue

        texts = df[field].fillna("").tolist()
        if field == "tasks":
            texts = [clean_tasks(t) for t in texts]

        logger.info("Embedding field '%s' (%d texts)...", field, len(texts))
        vecs = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
        )

        out_path = os.path.join(output_dir, f"{field}.csv")
        dim_cols = [f"dim_{i}" for i in range(vecs.shape[1])]
        pd.DataFrame(vecs, index=df["code"].values, columns=dim_cols).to_csv(out_path, index=True)
        logger.info("Saved embeddings to %s (shape %s)", out_path, vecs.shape)
        embeddings[field] = vecs

    return embeddings
