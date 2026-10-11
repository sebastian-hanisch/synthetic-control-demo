"""Orakel-Tests: der Kern gegen einen unabhängigen Rechenweg (ORACLE-PLAYBOOK §2).

1. Servicegrad ohne Trend und Schock gegen die exakte Formel der Order-up-to-Regel (unabhängig erzeugte Nachfrage).
2. Synthetische Kontrolle: nnls-Gewichte gegen eine allgemeine Optimierung (SLSQP) und gegen die KKT-Bedingungen.
3. Placebo-p-Wert gegen direkte Aufzählung; kleinster möglicher Wert 1 / (J + 1).
4. DiD ist erwartungstreu bei parallelen Verläufen und verzerrt um den Trend-Unterschied sonst (Simulation mit bekanntem Modell)."""

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import nbinom

import syn_constants as C
import syn_estimators as E
import syn_world as W


def fill_rate_formula(mean, disp, lead, z):
    """Exakter Servicegrad (Typ 2) der Order-up-to-Regel mit täglicher Prüfung und i.i.d. negativ-binomialer Nachfrage:
    1 - (E[(D_L - S)+] - E[(D_{L-1} - S)+]) / mean, D_n = NegBin(n * disp, Mittel n * mean)."""
    sd = np.sqrt(lead * (mean + mean**2 / disp))
    s = int(np.round(lead * mean + z * sd))

    def excess(n):
        if n == 0:
            return max(0.0, -s)
        nn, mm = n * disp, n * mean
        p = nn / (nn + mm)
        k = np.arange(s, s + 2000)
        return float(nbinom.sf(k, nn, p).sum())

    return 1.0 - (excess(lead) - excess(lead - 1)) / mean


def test_period_service_level_matches_the_exact_formula_without_trend_and_shock():
    rng = np.random.default_rng(2)
    worlds = 300
    mu = C.site_means()[[0, 3, 7]]
    total = C.BURN_DAYS + W.DAYS
    demand = np.empty((worlds, 3, total), dtype=np.int32)
    for s in range(3):
        demand[:, s, :] = nbinom.rvs(C.DISP, C.DISP / (C.DISP + mu[s]), size=(worlds, total), random_state=rng)
    for z in (1.0, 1.6):
        y = W.simulate_periods(demand, [0, 3, 7], z)
        for s in range(3):
            assert y[:, s, :].mean() == pytest.approx(fill_rate_formula(mu[s], C.DISP, C.LEAD, z), abs=0.004), (z, s)


def test_sc_weights_match_a_general_optimizer():
    rng = np.random.default_rng(6)
    c = rng.uniform(0.85, 0.98, (C.T0, 5))
    y = 0.4 * c[:, 1] + 0.3 * c[:, 3] + 0.3 * c[:, 4] + rng.normal(0, 0.01, C.T0)
    w = E.sc_weights(y, c)
    res = minimize(lambda v: ((y - c @ v) ** 2).sum(), np.full(5, 0.2), method="SLSQP", bounds=[(0, 1)] * 5,
                   constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1.0}], options={"ftol": 1e-14, "maxiter": 500})
    assert w.sum() == pytest.approx(1.0, abs=1e-5)
    assert ((y - c @ w) ** 2).sum() == pytest.approx(res.fun, rel=1e-4, abs=1e-9)
    assert w == pytest.approx(res.x, abs=2e-3)


def test_sc_weights_satisfy_the_kkt_conditions():
    rng = np.random.default_rng(7)
    c = rng.uniform(0.85, 0.98, (C.T0, 6))
    y = rng.uniform(0.85, 0.98, C.T0)
    w = E.sc_weights(y, c)
    grad = -2 * c.T @ (y - c @ w)
    lam = grad[w > 1e-6].mean()                           # Multiplikator der Summenbedingung
    assert np.allclose(grad[w > 1e-6], lam, atol=1e-3)
    assert (grad[w <= 1e-6] >= lam - 1e-3).all()


def test_placebo_pvalue_equals_counting():
    rng = np.random.default_rng(8)
    tau0 = rng.normal(size=20)
    taup = rng.normal(size=(20, 7))
    p = E.placebo_pvalue(tau0, taup)
    for i in range(20):
        count = sum(abs(t) >= abs(tau0[i]) for t in taup[i])
        assert p[i] == pytest.approx((1 + count) / 8)
    assert p.min() >= 1 / 8


def test_did_is_unbiased_under_parallel_paths_and_biased_by_the_trend_gap_otherwise():
    rng = np.random.default_rng(9)
    r, j = 4000, 6
    t = np.arange(C.N_PERIODS)
    common = rng.normal(0, 0.02, (r, 1, C.N_PERIODS))
    base = 0.9 + common + 0.001 * t[None, None, :]
    y0 = base[:, 0, :] + rng.normal(0, 0.02, (r, C.N_PERIODS)) + 0.03 * (t >= C.T0)
    yc = base + rng.normal(0, 0.02, (r, j, C.N_PERIODS))
    assert E.did(y0, yc).mean() == pytest.approx(0.03, abs=0.003)
    y0_own = y0 + 0.002 * t[None, :]                      # eigener Trend von 0,002 je Periode
    gap = 0.002 * ((C.T0 + C.N_PERIODS - 1) / 2 - (C.T0 - 1) / 2)
    assert E.did(y0_own, yc).mean() == pytest.approx(0.03 + gap, abs=0.003)
    assert E.did_trend(y0_own, yc).mean() == pytest.approx(0.03, abs=0.004)
