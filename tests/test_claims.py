"""Jede Zahl des README (Abschnitt "Befunde") wird hier nachgerechnet.

Die Zahlen stammen aus `precomputed_sweeps.json` (2 000 simulierte Welten je Zelle, SplitMix64, Seeds fest); zwei Zellen werden
zusätzlich mit denselben Funktionen wie in der App neu gerechnet, damit die Datei nicht von der Rechnung abweicht."""

import json
from pathlib import Path

import pytest

import syn_constants as C
import syn_evaluation as E

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cells():
    data = json.loads((ROOT / "precomputed_sweeps.json").read_text(encoding="utf-8"))
    return {(c["sweep"], c["scenario"], c["J"], round(c["zb"], 1)): c for c in data["cells"]}


def scen(cells, scenario, j):
    return cells[("scenario", scenario, j, 1.6)]


def effect(cells, j, zb):
    return cells[("scenario", "ruhig", j, 1.6)] if abs(zb - 1.6) < 1e-9 else cells[("effect", "ruhig", j, zb)]


def test_the_sweep_file_matches_a_fresh_computation(cells):
    for scenario, j in (("ruhig", 5), ("gemeinsam", 5)):
        fresh = E.summarize(2000, scenario, j, 1.6, tag=3)
        stored = scen(cells, scenario, j)
        assert fresh["truth"] == pytest.approx(stored["truth"], abs=0.01)
        for k in C.ESTIMATORS:
            for key in ("bias", "sd", "rmse"):
                assert fresh["rows"][k][key] == pytest.approx(stored["rows"][k][key], abs=0.03), (scenario, k, key)
        assert fresh["inference"]["cr_null_reject"] == pytest.approx(stored["inference"]["cr_null_reject"], abs=0.01)


def test_the_true_effect_per_scenario(cells):
    assert scen(cells, "ruhig", 12)["truth"] == pytest.approx(4.6, abs=0.1)
    for s in ("gemeinsam", "eigen", "verschieden"):
        assert scen(cells, s, 12)["truth"] == pytest.approx(8.0, abs=0.1), s
    assert effect(cells, 12, 1.2)["truth"] == pytest.approx(1.97, abs=0.05)
    assert effect(cells, 12, 2.0)["truth"] == pytest.approx(5.93, abs=0.05)


def test_in_a_calm_world_the_control_adds_nothing_for_did(cells):
    for j, naiv, did, sc in ((12, 2.31, 2.43, 2.14), (30, 2.33, 2.36, 2.14)):
        r = scen(cells, "ruhig", j)["rows"]
        assert r["naiv"]["rmse"] == pytest.approx(naiv, abs=0.05)
        assert r["did"]["rmse"] == pytest.approx(did, abs=0.05)
        assert r["sc"]["rmse"] == pytest.approx(sc, abs=0.05)
        for k in C.ESTIMATORS:
            assert abs(r[k]["bias"]) < 0.15
    assert scen(cells, "ruhig", 12)["rows"]["did_trend"]["rmse"] == pytest.approx(4.39, abs=0.08)


def test_a_common_trend_is_removed_by_did_and_sc(cells):
    r = scen(cells, "gemeinsam", 12)["rows"]
    assert r["naiv"]["bias"] == pytest.approx(-4.81, abs=0.1) and r["naiv"]["rmse"] == pytest.approx(6.10, abs=0.1)
    assert r["did"]["bias"] == pytest.approx(-0.06, abs=0.1) and r["did"]["rmse"] == pytest.approx(3.24, abs=0.08)
    assert r["sc"]["bias"] == pytest.approx(-0.15, abs=0.1) and r["sc"]["rmse"] == pytest.approx(3.05, abs=0.08)
    assert r["did_trend"]["rmse"] == pytest.approx(5.76, abs=0.1)


def test_an_own_trend_defeats_the_controls(cells):
    r = scen(cells, "eigen", 12)["rows"]
    assert r["naiv"]["bias"] == pytest.approx(-4.82, abs=0.1)
    assert r["did"]["bias"] == pytest.approx(-4.82, abs=0.1) and r["did"]["rmse"] == pytest.approx(5.83, abs=0.1)
    assert r["sc"]["bias"] == pytest.approx(-7.82, abs=0.1) and r["sc"]["rmse"] == pytest.approx(8.39, abs=0.1)
    assert r["did_trend"]["bias"] == pytest.approx(0.02, abs=0.3)
    assert r["did_trend"]["sd"] == pytest.approx(5.45, abs=0.1) and r["did_trend"]["rmse"] == pytest.approx(5.70, abs=0.1)


def test_different_trends_leave_a_large_part_of_the_bias(cells):
    r12, r30 = scen(cells, "verschieden", 12)["rows"], scen(cells, "verschieden", 30)["rows"]
    assert (r12["naiv"]["bias"], r12["did"]["bias"], r12["sc"]["bias"]) == pytest.approx((-4.86, -2.55, -2.86), abs=0.1)
    assert (r30["did"]["bias"], r30["sc"]["bias"]) == pytest.approx((-2.01, -2.12), abs=0.1)
    assert r12["did_trend"]["bias"] == pytest.approx(-0.03, abs=0.3) and r12["did_trend"]["rmse"] == pytest.approx(5.86, abs=0.1)


def test_cluster_robust_errors_fail_with_one_treated_site(cells):
    expected = {5: (32.6, 67), 12: (53.0, 47), 30: (70.5, 30)}
    for j, (null, cov) in expected.items():
        i = scen(cells, "ruhig", j)["inference"]
        assert 100 * i["cr_null_reject"] == pytest.approx(null, abs=1.0), j
        assert 100 * i["cr_coverage"] == pytest.approx(cov, abs=1.0), j
        assert 100 * i["iid_null_reject"] == pytest.approx({5: 5.7, 12: 5.2, 30: 5.1}[j], abs=0.6), j


def test_the_regression_with_iid_errors_breaks_with_an_own_trend(cells):
    for j, v in ((5, 40.3), (12, 42.9), (30, 46.0)):
        assert 100 * scen(cells, "eigen", j)["inference"]["iid_null_reject"] == pytest.approx(v, abs=1.5), j


def test_placebo_p_values_are_coarse(cells):
    r5 = scen(cells, "ruhig", 5)["rows"]
    for k in C.ESTIMATORS:
        assert r5[k]["p_obs"]["0.05"] == 0.0 and r5[k]["p_obs"]["0.1"] == 0.0
    assert 100 * r5["sc"]["p_obs"]["0.2"] == pytest.approx(74, abs=1.5)
    assert 100 * r5["sc"]["p_nul"]["0.2"] == pytest.approx(20, abs=1.5)
    r12 = scen(cells, "ruhig", 12)["rows"]
    assert r12["sc"]["p_obs"]["0.05"] == 0.0
    assert 100 * r12["sc"]["p_obs"]["0.1"] == pytest.approx(61, abs=1.5)
    assert 100 * r12["did"]["p_obs"]["0.1"] == pytest.approx(49, abs=1.5)
    r30 = scen(cells, "ruhig", 30)["rows"]
    assert 100 * r30["sc"]["p_obs"]["0.05"] == pytest.approx(46, abs=1.5)
    assert 100 * r30["did"]["p_obs"]["0.05"] == pytest.approx(34, abs=1.5)
    assert 100 * r30["sc"]["p_obs"]["0.1"] == pytest.approx(72, abs=1.5)
    for k in C.ESTIMATORS:
        assert 2.0 <= 100 * r30[k]["p_nul"]["0.05"] <= 6.0 and 8.0 <= 100 * r30[k]["p_nul"]["0.1"] <= 12.0


def test_the_placebo_test_under_an_own_trend_cries_wolf_for_sc(cells):
    r = scen(cells, "eigen", 12)["rows"]
    assert 100 * r["sc"]["p_nul"]["0.1"] == pytest.approx(85, abs=1.5)
    assert 100 * r["did_trend"]["p_nul"]["0.1"] == pytest.approx(13, abs=1.5)
    r30 = scen(cells, "eigen", 30)["rows"]
    assert 100 * r30["sc"]["p_nul"]["0.05"] == pytest.approx(79, abs=1.5)
    assert 100 * r30["did"]["p_nul"]["0.05"] == pytest.approx(36, abs=1.5)


def test_power_by_effect_size_with_30_controls(cells):
    expected = {1.2: 22, 1.4: 46, 1.6: 72, 1.8: 83, 2.0: 93}
    for zb, v in expected.items():
        r = effect(cells, 30, zb)["rows"]["sc"]
        assert 100 * r["p_obs"]["0.1"] == pytest.approx(v, abs=1.5), zb


def test_synthetic_control_fit_before_the_start(cells):
    for j, v in ((5, 5.12), (12, 4.34), (30, 3.66)):
        assert scen(cells, "ruhig", j)["sc"]["rmspe_pre"] == pytest.approx(v, abs=0.1), j
    assert scen(cells, "ruhig", 30)["sc"]["n_weights"] == pytest.approx(6.4, abs=0.2)
    assert scen(cells, "ruhig", 30)["sc"]["max_weight"] == pytest.approx(0.36, abs=0.03)


def test_default_seed_world():
    r = E.run_one(C.SCENARIOS[2], 12, C.DEFAULT_ZB, C.DEFAULT_SEED)
    assert 100 * r["truth"] == pytest.approx(7.27, abs=0.02)
    est = {k: 100 * v for k, v in r["est"].items()}
    assert est["naiv"] == pytest.approx(2.89, abs=0.03)
    assert est["did"] == pytest.approx(7.07, abs=0.03)
    assert est["did_trend"] == pytest.approx(7.78, abs=0.03)
    assert est["sc"] == pytest.approx(7.66, abs=0.03)
    assert r["p"]["sc"] == pytest.approx(1 / 13, abs=1e-9) and r["p"]["did"] == pytest.approx(3 / 13, abs=1e-9)
