import logging
import warnings


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for _noisy in ("httpx", "sentence_transformers", "huggingface_hub", "transformers"):
        logging.getLogger(_noisy).setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore",
        message="n_jobs value.*overridden",
        category=UserWarning,
        module="umap",
    )
