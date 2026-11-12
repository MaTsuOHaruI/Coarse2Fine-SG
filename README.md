# Soil Grain Size Distributions from Smartphone Photos

Computer Vision final project, National Sun Yat-sen University.
Team: Haruki Matsuo, Erhans Arya Kencana, Jason Supala.

Task: the Kaggle competition [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos). Given top-down photos of a soil sample in a tray, predict the cumulative mass percentage passing eleven sieve sizes (0.002 mm to 200 mm). The metric is a log-weighted Earth Mover's Distance (lower is better).

Idea ("See the Coarse, Learn the Fine"): measure the grains the camera can resolve with a frozen segmentation model (SAM), learn the rest of the curve with a small regularised head, and blend the two according to how much coarse evidence was measured.

## Status

| Part | State |
|---|---|
| Metric, data loaders, photo scale, leave-one-soil-out harness | working, tested |
| Constant and mean-curve baselines, submission writer with validity checks | working, tested |
| Path A: SAM (tiled) or classical segmentation, grain measurement, coarse curve | working; SAM not yet run on the full data |
| Path B: texture, colour and embedding features, PCA-composition ridge head | working, tested |
| Path C: evidence-weighted stitching | working, tested |
| Tray detection, clod filter, nested cross-validation, DINOv2 runs | to do |

Leave-one-soil-out on the 24 training soils (classical segmenter, colour thumbnail instead of DINOv2):

| Method | EMD |
|---|---|
| Constant baseline (organisers) | 102.0 |
| Mean training curve | 89.9 |
| Path B only (ridge) | 42.8 |
| Path B + stitch, adaptive weight / hard | 43.7 / 43.7 |

Hyper-parameters were not tuned in a nested loop yet, so these are optimistic by a few points.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/MaTsuOHaruI/Coarse2Fine-SG.git
cd Coarse2Fine-SG
uv sync
```

Without uv: `python -m venv .venv`, activate it, then `pip install -e . pytest` (or `pip install -r requirements.txt` for the core packages only, without SAM / DINOv2).

### Data

Download the competition data from Kaggle and unzip it into `dataset/` in the repository root, so that it looks like this:

```
dataset/
    Training_labels_updated.csv
    ppm_updated.csv
    sample_submission.csv
    Training-All_Photos_updated/Training-All_Photos_updated/*.jpg
    Test_All_Photos/Test_All_Photos/*.JPG
```

`dataset/`, `data/` (external datasets), `cache/` and `runs/` are not tracked by git. To keep the data somewhere else, set `SOILGSD_DATA_DIR` (and `SOILGSD_CACHE_DIR` for cached masks, features and the SAM checkpoint).

## Running

Quick commands (print to the terminal):

```bash
uv run python -m soilgsd demo                    # synthetic soils, LOSO-CV, no data needed
uv run python -m soilgsd check                   # file name -> (phone, sample_id) and px/mm for every photo
uv run python -m soilgsd loso                    # LOSO-CV of the pipeline
uv run python -m soilgsd predict --out sub.csv   # add --sam-ckpt sam_vit_b.pth for SAM
```

Scripts (write a run folder under `runs/` for the results log):

```bash
uv run pytest                                          # tests (data tests are skipped without dataset/)
uv run python scripts/eval_baselines.py                # the two reference scores, 102.0 and 89.9
uv run python scripts/scale_table.py                   # photo, soil, phone, px/mm, crossover sieve
uv run python scripts/make_cache.py                    # masks + features of every photo into cache/ (resumable)
uv run python scripts/run_loso.py                      # LOSO-CV of mean curve, Path B, stitched, hard stitch
uv run python scripts/make_submission.py --model pure_b   # checked submission CSV (mean_curve | pure_b | adaptive | hard)
```

Every script accepts `--config <yaml>` (default `configs/default.yaml`), `--device auto|cuda|cpu` and `--limit N`. The pipeline scripts also take `--segmenter classical|sam` and `--dino`. Settings in `configs/default.yaml` override the defaults in `soilgsd/config.py`.

## Running on Colab

GPU steps (SAM masks, DINOv2 embeddings) are meant for a Colab T4.

1. Upload the `dataset/` folder to Google Drive at `MyDrive/soilgsd/dataset/`.
2. Open `notebooks/colab_runner.ipynb` in Colab and select a T4 GPU runtime.
3. Run the cells top to bottom (`REPO_URL` already points at this repository).

The notebook mounts Drive, clones or updates the repository, installs the package (keeping Colab's own PyTorch), downloads the SAM checkpoint once into the Drive cache, and runs the scripts. Masks and features are cached in `MyDrive/soilgsd/cache/`, so a stopped session continues where it left off, and the cache can be downloaded for CPU-only work on a laptop.

## Code map

| Module | Proposal |
|---|---|
| `config.py` | sieves, hyper-parameters, paths, seeds, shared script flags |
| `crossover.py` | Eq.(2) adaptive d_c, true scale rho |
| `segment.py` | SAM (tiled) / ClassicalSegmenter fallback, mask cache |
| `path_a.py` | Path A: Feret b_g, m_g ∝ a·b^p, G_coarse (Eq.3), n_eff |
| `features.py` | Path B inputs (i)-(iv); `Dinov2Embedder` is the DINOv2 hook |
| `path_b.py` | PCA-composition head (Eq.4), ridge |
| `path_c.py` | Eq.(5) blend; modes adaptive / pure_b (w=0) / hard (w=1) |
| `pipeline.py` | per-soil extraction (cached) and the fitted model |
| `cv.py`, `metrics.py` | LOSO-CV, Eq.(1) EMD |
| `baselines.py` | constant and mean-curve baselines |
| `submit.py` | submission writer with validity checks |
| `data.py` | Kaggle loader; file name -> (phone, sample_id) is inferred and verified by `check` |
| `synth.py` | synthetic soils with exact labels |

Other folders: `scripts/` (one script per step), `configs/` (YAML), `tests/`, `notebooks/` (Colab runner), `results/results.csv` (one row per reported run), `docs/` (proposal).

## Rules we follow

- Cross-validation is always grouped by soil; photos of one soil never straddle a split.
- Anything fitted on data is fitted inside the training fold only.
- Model selection uses cross-validation, not the leaderboard.
- Every submission file goes through the validity checks in `submit.py`.

## External code and data

- Segment Anything: https://github.com/facebookresearch/segment-anything
- DINOv2: https://github.com/facebookresearch/dinov2
- GRAI1 release (Soranzo 2025): https://doi.org/10.5281/zenodo.14725633
- FRDR Photogranulometry: https://doi.org/10.20383/103.01316
- SandSnap: https://doi.org/10.5281/zenodo.7063226
