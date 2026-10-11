"""Auswertung: Wahrheit je Welt, Verhalten der Schätzer je Szenario, Reproduzierbarkeit."""

import numpy as np
import pytest

import syn_constants as C
import syn_evaluation as E

REPS = 120


def test_truth_is_zero_without_a_new_rule_and_positive_with_it():
    y_obs, y_nul, _ = E.simulate_worlds(40, "ruhig", 5, C.Z_A, 1)
    assert np.array_equal(y_obs, y_nul)
    y_obs, y_nul, _ = E.simulate_worlds(40, "ruhig", 5, 1.6, 1)
    assert np.array_equal(y_obs[:, :C.T0], y_nul[:, :C.T0])
    assert (y_obs - y_nul)[:, C.T0:].mean() > 0.03


def test_the_observed_world_differs_from_the_null_world_only_after_the_start():
    y_obs, y_nul, yc = E.simulate_worlds(30, "gemeinsam", 5, 1.6, 2)
    assert yc.shape == (30, 5, C.N_PERIODS)
    assert np.array_equal(y_obs[:, :C.T0], y_nul[:, :C.T0])
    assert not np.array_equal(y_obs[:, C.T0:], y_nul[:, C.T0:])


def test_a_common_trend_biases_before_after_but_not_did():
    s = E.summarize(REPS, "gemeinsam", 5, 1.6)["rows"]
    assert s["naiv"]["bias"] < -3.0
    assert abs(s["did"]["bias"]) < 1.0 and abs(s["sc"]["bias"]) < 1.0


def test_an_own_trend_biases_did_and_sc_and_only_the_pretrend_adjustment_helps():
    s = E.summarize(REPS, "eigen", 5, 1.6)["rows"]
    assert s["did"]["bias"] < -3.0
    assert s["sc"]["bias"] < s["did"]["bias"] < -3.0
    assert abs(s["did_trend"]["bias"]) < 1.6
    assert s["did_trend"]["sd"] > s["did"]["sd"] + 1.0


def test_without_changes_all_estimators_are_unbiased():
    s = E.summarize(REPS, "ruhig", 5, 1.6)["rows"]
    for k in C.ESTIMATORS:
        assert abs(s[k]["bias"]) < 1.0, k


def test_the_smallest_placebo_p_value_is_one_over_j_plus_one():
    s5 = E.summarize(60, "ruhig", 5, 1.6)["rows"]
    for k in C.ESTIMATORS:
        assert s5[k]["p_obs"]["0.1"] == 0.0 and s5[k]["p_nul"]["0.05"] == 0.0     # 1/6 > 0,1
    s12 = E.summarize(60, "ruhig", 12, 1.6)["rows"]
    assert all(s12[k]["p_obs"]["0.05"] == 0.0 for k in C.ESTIMATORS)             # 1/13 > 0,05


def test_summaries_are_reproducible_and_chunk_independent():
    a = E.summarize(50, "schock", 5, 1.6)
    b = E.summarize(50, "schock", 5, 1.6)
    assert a["rows"] == b["rows"] and a["truth"] == b["truth"]
    assert E.summarize(50, "schock", 5, 1.6, seed=1)["truth"] != a["truth"]


def test_run_one_returns_series_and_estimates():
    r = E.run_one("gemeinsam", 5, 1.6, 35)
    assert r["y0"].shape == (C.N_PERIODS,) and r["yc"].shape == (5, C.N_PERIODS)
    assert r["w"].sum() == pytest.approx(1.0, abs=1e-5)
    assert set(r["est"]) == set(C.ESTIMATORS) and all(0 < v <= 1 for v in r["p"].values())
    assert r["placebo"]["did"].shape == (5,)
