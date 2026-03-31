# Occupational Similarity

Replication code for the occupational similarity and occupational vector measures introduced in:

> Dabed, D., Genz, S., & Rademakers, E. (2025). Equalising the effects of automation? The role of task overlap for job finding. *Labour Economics*, 96, 102766. https://doi.org/10.1016/j.labeco.2025.102766

## What this does

The pipeline generates text embeddings and pairwise similarity measures for all occupations in the [ISCO-08](https://www.ilo.org/public/english/bureau/stat/isco/isco08/) classification, using four text fields per occupation: **title**, **definition**, **tasks**, and **included occupations**.

Steps:

1. **Download** the ISCO-08 structure and definitions file from the ILO website (falls back to a local copy if unavailable).
2. **Parse** occupation codes, titles, definitions, task descriptions, and included occupations.
3. **Embed** each text field using the [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) sentence transformer (768 dimensions).
4. **Compute** pairwise cosine similarity matrices between all occupations for each text field.
5. **Reduce** the 768-dimensional embeddings to 2D via Kernel PCA (RBF kernel) and UMAP (cosine metric).
6. **Visualise** the UMAP projections in a 2×2 plot coloured by ISCO major group.

## Outputs

| Path | Description |
|---|---|
| `outputs/embeddings/{field}.csv` | 768-dim embedding vectors (one row per occupation) |
| `outputs/similarity/{field}.csv` | Pairwise cosine similarity matrix |
| `outputs/reduced/{field}_umap.csv` | 2D UMAP coordinates |
| `outputs/reduced/{field}_kpca.csv` | 2D Kernel PCA coordinates |
| `outputs/figures/umap_2x2.png` | UMAP visualisation |

## Difference from the paper

The measures produced here are **not identical** to those used in the paper. The paper relies on administrative data from the Public Employment Service of Flanders, Belgium (VDAB), which is not publicly available. That dataset contains 1,157 distinct jobs defined by the Belgian PES classification (derived from the French ROME-v3 system), each further split into up to three experience levels (less than two years, two to five years, and more than five years), yielding a finer occupational grid than any international standard classification. Task descriptions in the paper come from the COMPETENT matrix — a structured inventory of 10,303 unique tasks compiled by the Belgian authorities from expert observations and surveys.

This repository re-implements the same methodology using the publicly available [ISCO-08](https://www.ilo.org/public/english/bureau/stat/isco/isco08/) classification from the ILO, which contains approximately 500 unit groups with no experience dimension. The resulting similarity measures therefore differ in granularity and coverage, but the underlying approach — sentence embeddings of occupational text fields and cosine similarity — is identical.

## Usage

```bash
pip install -r requirements.txt
python pipeline.py
```

Results are fully reproducible (seed 420 fixed throughout).

## Citation

If you use this code or the resulting measures, please cite:

```bibtex
@article{dabed2025equalising,
  title   = {Equalising the effects of automation? {The} role of task overlap for job finding},
  author  = {Dabed, Diego and Genz, Sabrina and Rademakers, Emilie},
  journal = {Labour Economics},
  volume  = {96},
  pages   = {102766},
  year    = {2025},
  doi     = {10.1016/j.labeco.2025.102766}
}
```
