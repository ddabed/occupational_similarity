import logging
import os

import requests

ILO_URL = (
    "https://www.ilo.org/ilostat-files/ISCO/newdocs-08-2021/ISCO-08/"
    "ISCO-08%20EN%20Structure%20and%20definitions.xlsx"
)
FALLBACK_PATH = "ISCO-08-en.xlsx"

logger = logging.getLogger(__name__)


def download_isco(dest: str = "data/ISCO-08-en.xlsx") -> str:
    """Download ISCO-08 Excel file from ILO. Falls back to local file on failure.

    Args:
        dest: Destination path for the downloaded file.

    Returns:
        Path to the Excel file (downloaded or fallback).

    Raises:
        FileNotFoundError: If download fails and no fallback file exists.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        logger.info("Downloading ISCO-08 data from ILO...")
        response = requests.get(ILO_URL, timeout=60)
        response.raise_for_status()
        with open(dest, "wb") as f:
            f.write(response.content)
        logger.info("Downloaded to %s", dest)
        return dest
    except Exception as exc:
        logger.warning("Download failed (%s). Trying fallback: %s", exc, FALLBACK_PATH)

    if os.path.exists(FALLBACK_PATH):
        logger.info("Using fallback file: %s", FALLBACK_PATH)
        return FALLBACK_PATH

    raise FileNotFoundError(
        f"Download failed and fallback file '{FALLBACK_PATH}' not found. "
        "Please place the ISCO-08-en.xlsx file in the project root."
    )
