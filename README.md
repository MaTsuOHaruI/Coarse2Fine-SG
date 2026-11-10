# soilgsd — MVP of "See the Coarse, Learn the Fine"

    pip install numpy scipy scikit-learn opencv-python scikit-image pandas
    python -m soilgsd demo                                   # synthetic soils, LOSO-CV, no data needed
    python -m soilgsd check                  # run inside the Kaggle data parent folder (or --root DIR)
    python -m soilgsd loso
    python -m soilgsd predict --out sub.csv  # add --sam-ckpt sam_vit_b.pth for SAM

| Module | Proposal |
|---|---|
| `crossover.py` | Eq.(2) adaptive d_c, true scale rho |
| `segment.py` | SAM (tiled) / ClassicalSegmenter fallback |
| `path_a.py` | Path A: Feret b_g, m_g∝a·b^p, G_coarse (Eq.3), n_eff |
| `features.py` | Path B inputs (i)-(iv); `Dinov2Embedder` is the DINOv2 hook |
| `path_b.py` | PCA-composition head (Eq.4), ridge |
| `path_c.py` | Eq.(5) blend; modes adaptive / pure_b (w=0) / hard (w=1) |
| `cv.py`, `metrics.py` | LOSO-CV, Eq.(1) EMD |
| `synth.py` | synthetic soils with exact labels |

`data.py` reads the Kaggle layout directly; file name -> (phone, sample_id) is inferred and verified by `check`.
