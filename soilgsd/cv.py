"""Leave-one-soil-out cross-validation. Splits are by soil, never by photo.

Anything passed in as `fit_predict` may only learn from the training-fold table it is
given: PCA bases, scalers and regressors are fitted inside that function.
"""
import numpy as np
import pandas as pd
from .config import Config, ID_COLUMN, SIEVE_COLUMNS
from .metrics import emd
from .baselines import mean_curve


def leave_one_soil_out(labels: pd.DataFrame, fit_predict, form: str = "left") -> pd.DataFrame:
    """Hold out each soil once and score the prediction made without it.

    labels: one row per soil, sample_id + the eleven sieve columns.
    fit_predict(train_labels, held_out_id) -> 11 percentages; train_labels excludes the held-out soil.
    Out: one row per soil: sample_id, emd, and the predicted curve. Mean of `emd` is the CV score.
    """
    if labels[ID_COLUMN].duplicated().any():
        raise ValueError("labels must have exactly one row per soil")
    rows = []
    for held_out_id in labels[ID_COLUMN]:
        is_held_out = labels[ID_COLUMN] == held_out_id
        train_labels = labels[~is_held_out].reset_index(drop=True)
        true_curve = labels.loc[is_held_out, SIEVE_COLUMNS].to_numpy(dtype=float)[0]
        predicted = np.asarray(fit_predict(train_labels, held_out_id), dtype=float)
        row = {ID_COLUMN: held_out_id, "emd": emd(true_curve, predicted, form=form)}
        row.update(dict(zip(SIEVE_COLUMNS, predicted)))
        rows.append(row)
    return pd.DataFrame(rows)


def loso(extracted, curves, cfg=Config()):
    """LOSO-CV of the full pipeline. Returns {method: (mean EMD, per-soil EMDs)}.

    Methods: mean_curve (arithmetic mean of the training curves; the MVP used the clr centre,
    which scores 92.1 instead of the reference 89.9), pure_b (w = 0), adaptive, hard (w = 1).
    """
    from .pipeline import SoilGSD                      # local import: pipeline imports cv's neighbours
    ids = [x.sample_id for x in extracted]
    by_id = {x.sample_id: x for x in extracted}
    labels = pd.DataFrame(np.asarray(curves, float), columns=SIEVE_COLUMNS)
    labels.insert(0, ID_COLUMN, ids)

    def train_xc(train):
        return [by_id[s] for s in train[ID_COLUMN]], train[SIEVE_COLUMNS].to_numpy(float)

    def model_predictor(mode):
        def fp(train, held):
            xs, cs = train_xc(train)
            return SoilGSD(cfg).fit(xs, cs).predict(by_id[held], mode)["final"]
        return fp

    predictors = {"mean_curve": lambda train, held: mean_curve(train[SIEVE_COLUMNS].to_numpy(float))}
    predictors.update({m: model_predictor(m) for m in ("pure_b", "adaptive", "hard")})
    res = {}
    for k, fp in predictors.items():
        e = leave_one_soil_out(labels, fp)["emd"].tolist()
        res[k] = (float(np.mean(e)), e)
    return res


def nested_leave_one_soil_out(*args, **kwargs):
    """Outer LOSO with an inner LOSO for hyperparameters."""
    raise NotImplementedError("TODO: nested leave-one-soil-out")


def paired_bootstrap(*args, **kwargs):
    """Paired bootstrap interval over soils for the EMD difference of two methods."""
    raise NotImplementedError("TODO: paired bootstrap")
