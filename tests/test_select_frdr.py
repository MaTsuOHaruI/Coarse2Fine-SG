"""Tests for scripts/select_frdr_soils.py (no data needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_path = Path(__file__).resolve().parents[1] / "scripts" / "select_frdr_soils.py"
_spec = importlib.util.spec_from_file_location("select_frdr_soils", _path)
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

ASTM_MM = np.array([0.075, 0.425, 2.0, 4.75, 9.5, 19.0])


def test_passing_at_hits_label_on_a_sieve():
    p = np.array([5, 20, 40, 60, 85, 100.0])
    assert abs(sel.passing_at(ASTM_MM, p, 2.0) - 40.0) < 1e-9


def test_passing_at_between_sieves_stays_between_neighbours():
    p = np.array([5, 20, 40, 60, 85, 100.0])
    v = sel.passing_at(ASTM_MM, p, 6.3)
    assert 60.0 < v < 85.0


def test_passing_at_outside_range():
    p = np.array([5, 20, 40, 60, 85, 100.0])
    assert sel.passing_at(ASTM_MM, p, 50.0) == 100.0  # curve already at 100
    p_open = np.array([5, 20, 40, 60, 85, 95.0])
    assert np.isnan(sel.passing_at(ASTM_MM, p_open, 50.0))  # top unknown
    assert np.isnan(sel.passing_at(ASTM_MM, p, 0.01))


def test_plan_counts_moves_shortfall_to_neighbour():
    # bin 1 (10 to 30) has only 3 soils: its 6 missing go to the closest bin with spare soils
    counts = sel.plan_counts([10, 3, 20, 20, 20])
    assert counts[1] == 3
    assert sum(counts) == sum(n for _, _, n in sel.COARSE_BINS)
    assert counts[2] == 9 + 6


def test_plan_counts_never_exceeds_available():
    available = [1, 2, 0, 5, 3]
    counts = sel.plan_counts(available)
    assert all(c <= a for c, a in zip(counts, available))
    assert sum(counts) == sum(available)


def test_selection_is_deterministic():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(120):
        p = np.sort(rng.uniform(0, 100, len(ASTM_MM)))
        p[-1] = 100.0
        rows.append({"soil_id": f"S{k:03d}", **{str(s): v for s, v in zip(ASTM_MM, p)}})
    table = sel.coarse_table(pd.DataFrame(rows), "soil_id")
    table["bin"] = [sel.assign_bin(c, p) for c, p in zip(table["coarse_pct"], table["p_6_3mm_pct"])]
    a, _, _ = sel.select_soils(table, "soil_id", seed=0)
    b, _, _ = sel.select_soils(table, "soil_id", seed=0)
    assert list(a["soil_id"]) == list(b["soil_id"])


def test_soil_without_6_3mm_value_is_not_eligible():
    sizes = np.array([0.075, 0.425, 2.0, 4.75])
    labels = pd.DataFrame([{"soil_id": "0078", **{str(s): v for s, v in zip(sizes, [5, 20, 40, 90])}}])
    table = sel.coarse_table(labels, "soil_id")
    row = table.iloc[0]
    assert row["coarse_pct"] == 60.0
    assert sel.assign_bin(row["coarse_pct"], row["p_6_3mm_pct"]) == -1


def _fake_frdr_csv(path):
    """Two samples: 0001 with 4 dry trays (3 images each) + moist images, 0002 with 1 dry tray."""
    sieves = ["80", "56", "40", "31.5", "28", "20", "14", "10", "5", "2.5", "2", "1.25", "0.63", "0.315", "0.16", "0.08"]
    curve = [100, 100, 100, 100, 100, 100, 95, 90, 70, 55, 50, 40, 30, 20, 10, 5]
    rows = []
    for sid, kind, trays in (("0001", "bulk", [11, 12, 13, 14]), ("0002", "splitspoon", [20])):
        for state in ("dry", "moist"):
            for tray in trays:
                for idx in (1, 2, 3):
                    row = {"sample_id": sid, "dry_or_moist": state, "sample_mass (g)": 1000.0, "%_water": 0.0,
                           "image_path": f"photogranulometry_v1_0/images_soil/{sid}_{kind}_{state}_tray{tray:03d}_{idx:02d}.jpg"}
                    row.update({f"%_{s}mm": v for s, v in zip(sieves, curve)})
                    rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_frdr_keeps_ids_and_state(tmp_path):
    csv = tmp_path / "frdr.csv"
    _fake_frdr_csv(csv)
    labels, photos = sel.load_frdr(csv, "dry")
    assert list(labels["sample_id"]) == ["0001", "0002"]
    assert "2" in labels.columns and "%_water" not in labels.columns
    assert labels.loc[0, "2"] == 50
    assert (photos["image_path"].str.contains("_dry_")).all()
    assert photos.groupby("sample_id")["tray"].nunique().to_dict() == {"0001": 4, "0002": 1}


def test_select_photos_one_image_per_tray(tmp_path):
    csv = tmp_path / "frdr.csv"
    _fake_frdr_csv(csv)
    _, photos = sel.load_frdr(csv, "dry")
    chosen = sel.select_photos(photos, ["0001"], n_trays=4, seed=0)
    assert len(chosen) == 4
    assert chosen["tray"].nunique() == 4
    again = sel.select_photos(photos, ["0001"], n_trays=4, seed=0)
    assert list(chosen["image_path"]) == list(again["image_path"])
