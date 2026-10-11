"""Schätzer und Placebo-Inferenz: von Hand an Mini-Instanzen und als Eigenschaften."""

import numpy as np
import pytest

import syn_constants as C
import syn_estimators as E

T0, T = C.T0, C.N_PERIODS


def test_naive_and_did_by_hand(flat_world):
    y0, yc = flat_world
    assert E.naive(y0) == pytest.approx([0.05] * 3)
    assert E.did(y0, yc) == pytest.approx([0.03] * 3)


def test_did_with_pretrend_removes_a_linear_drift_in_the_gap():
    t = np.arange(T, dtype=float)
    y0 = (0.90 + 0.001 * t)[None, :].copy()
    yc = np.full((1, 2, T), 0.92)
    y0[:, T0:] += 0.03                                   # wahre Wirkung 0,03 auf einem Vortrend im Abstand
    assert E.did(y0, yc)[0] == pytest.approx(0.03 + 0.001 * ((T0 + T - 1) / 2 - (T0 - 1) / 2))     # DiD nimmt den Trend mit
    assert E.did_trend(y0, yc)[0] == pytest.approx(0.03)


def test_sc_weights_recover_an_exact_mix():
    rng = np.random.default_rng(0)
    c = rng.uniform(0.85, 0.97, (T0, 3))
    y = 0.5 * c[:, 0] + 0.5 * c[:, 1]
    w = E.sc_weights(y, c)
    assert w == pytest.approx([0.5, 0.5, 0.0], abs=1e-5)
    assert w.sum() == pytest.approx(1.0, abs=1e-5) and (w >= 0).all()


def test_sc_effect_is_the_post_period_gap_to_the_weighted_controls():
    rng = np.random.default_rng(1)
    c = rng.uniform(0.85, 0.97, (1, 3, T))
    y0 = (0.25 * c[0, 0] + 0.75 * c[0, 2])[None, :].copy()
    y0[:, T0:] += 0.04
    tau, w, rm_pre, rm_post = E.sc(y0, c)
    assert tau[0] == pytest.approx(0.04, abs=1e-4)
    assert w[0] == pytest.approx([0.25, 0.0, 0.75], abs=1e-4)
    assert rm_pre[0] < 1e-4 and rm_post[0] == pytest.approx(0.04, abs=1e-4)


def test_placebo_pvalue_by_hand():
    tau0 = np.array([0.05])
    taup = np.array([[0.01, -0.06, 0.02]])
    assert E.placebo_pvalue(tau0, taup)[0] == pytest.approx((1 + 1) / 4)
    assert E.placebo_pvalue(np.array([0.5]), taup)[0] == pytest.approx(1 / 4)            # kleinster möglicher Wert 1 / (J + 1)
    assert E.placebo_pvalue(np.array([0.0]), taup)[0] == pytest.approx(1.0)


def test_placebo_taus_are_zero_for_identical_controls(flat_world):
    _, yc = flat_world
    plc = E.placebo_taus(yc)
    for k in C.ESTIMATORS:
        assert np.allclose(plc[k][:, 0], 0.02 if k == "naiv" else 0.0, atol=1e-9), k
        assert plc[k].shape == (3, 2)


def test_estimate_all_has_every_estimator_and_the_weights_sum_to_one(flat_world):
    y0, yc = flat_world
    out = E.estimate_all(y0, yc)
    assert set(C.ESTIMATORS) <= set(out)
    assert out["_w"].sum(axis=1) == pytest.approx([1.0] * 3, abs=1e-5)


def test_twfe_equals_the_regression_with_dummies():
    rng = np.random.default_rng(3)
    r, j = 4, 5
    y0 = rng.normal(0.9, 0.03, (r, T))
    yc = rng.normal(0.9, 0.03, (r, j, T))
    tau, se_iid, _, df = E.twfe(y0, yc)
    s = j + 1
    for i in range(r):
        y = np.vstack([y0[i], yc[i]]).ravel()
        unit = np.repeat(np.arange(s), T)
        time = np.tile(np.arange(T), s)
        d = ((unit == 0) & (time >= T0)).astype(float)
        x = np.column_stack([d, np.ones(s * T)] + [(unit == u).astype(float) for u in range(1, s)]
                            + [(time == t).astype(float) for t in range(1, T)])
        coef, res, *_ = np.linalg.lstsq(x, y, rcond=None)
        sigma2 = ((y - x @ coef) ** 2).sum() / (len(y) - x.shape[1])
        se = np.sqrt(sigma2 * np.linalg.inv(x.T @ x)[0, 0])
        assert tau[i] == pytest.approx(coef[0], abs=1e-9)
        assert se_iid[i] == pytest.approx(se, rel=1e-8)
    assert df == s * T - (s + T)


def test_twfe_estimate_equals_the_four_cell_did_for_a_balanced_panel():
    rng = np.random.default_rng(4)
    y0 = rng.normal(0.9, 0.03, (6, T))
    yc = rng.normal(0.9, 0.03, (6, 4, T))
    assert E.twfe(y0, yc)[0] == pytest.approx(E.did(y0, yc), abs=1e-10)
