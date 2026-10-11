"""Das Vehikel: Nachfrage mit Schock und Trend, Perioden-Servicegrad, Reproduzierbarkeit."""

import numpy as np
import pytest

import syn_constants as C
import syn_world as W


def test_trend_matrix_by_hand():
    tm = W.trend_matrix(np.array([0.0, 0.2]))
    assert tm.shape == (2, W.DAYS)
    assert tm[0].tolist() == [1.0] * W.DAYS
    assert tm[1, 0] == pytest.approx(1.0) and tm[1, -1] == pytest.approx(1.2)
    assert tm[1, W.DAYS // 2] == pytest.approx(1.0 + 0.2 * (W.DAYS // 2) / (W.DAYS - 1))


def test_scenario_trends_and_shocks():
    assert (W.scenario_trends("ruhig") == 0).all()
    assert (W.scenario_trends("gemeinsam") == C.TREND_G).all()
    eigen = W.scenario_trends("eigen")
    assert eigen[C.TREATED] == C.TREND_G and eigen.sum() == C.TREND_G
    verschieden = W.scenario_trends("verschieden")
    assert verschieden[C.TREATED] == C.TREND_G and verschieden.min() >= -0.10 - 1e-9 and verschieden.max() <= 0.30 + 1e-9
    assert W.scenario_shock("ruhig") == 0.0 and W.scenario_shock("schock") == C.SHOCK_SD


def test_z_schedule_switches_at_the_start_of_the_post_period():
    z = W.z_schedule(1.6)
    assert z.shape == (W.DAYS,)
    assert (z[: C.T0 * C.PERIOD_DAYS] == C.Z_A).all() and (z[C.T0 * C.PERIOD_DAYS:] == 1.6).all()


def test_demand_follows_size_weekday_and_trend():
    d = W.generate_demand(300, [0, C.TREATED], "eigen", 5)
    mu = C.site_means()
    dow = C.dow_factors()
    cell = d[:, 1, C.BURN_DAYS:]                                       # behandelter Standort, wächst um 20 %
    first_week = cell[:, :7].mean(axis=0)
    last_week = cell[:, -7:].mean(axis=0)
    assert first_week.mean() == pytest.approx(mu[C.TREATED] * 1.0, rel=0.05)
    assert last_week.mean() / first_week.mean() == pytest.approx(1.2, rel=0.08)
    assert d[:, 0, C.BURN_DAYS:C.BURN_DAYS + 7].mean(axis=0) == pytest.approx(mu[0] * dow, rel=0.1)


def test_shocks_are_common_across_sites():
    d = W.generate_demand(200, [0, 5, 9], "schock", 7)
    weekly = d[:, :, C.BURN_DAYS:C.BURN_DAYS + 28 * 7].reshape(200, 3, 28, 7).sum(axis=3).astype(float)
    weekly = weekly / weekly.mean(axis=(0, 2), keepdims=True)
    corr = np.corrcoef(weekly[:, 0, :].ravel(), weekly[:, 1, :].ravel())[0, 1]
    calm = W.generate_demand(200, [0, 5, 9], "ruhig", 7)
    calm_w = calm[:, :, C.BURN_DAYS:C.BURN_DAYS + 28 * 7].reshape(200, 3, 28, 7).sum(axis=3).astype(float)
    calm_w = calm_w / calm_w.mean(axis=(0, 2), keepdims=True)
    assert corr > 0.3 > abs(np.corrcoef(calm_w[:, 0, :].ravel(), calm_w[:, 1, :].ravel())[0, 1])


def test_demand_is_reproducible_and_independent_of_the_total_number_of_worlds():
    a = W.generate_demand(100, [3, 4], "schock", 11)
    b = W.generate_demand(250, [3, 4], "schock", 11)
    assert np.array_equal(a, b[:100])
    assert not np.array_equal(a, W.generate_demand(100, [3, 4], "schock", 12))


def test_period_service_levels_are_shares_and_higher_factor_is_never_worse():
    d = W.generate_demand(30, [C.TREATED], "ruhig", 3)
    low = W.simulate_periods(d, [C.TREATED], 1.0)
    high = W.simulate_periods(d, [C.TREATED], 1.6)
    assert low.shape == (30, 1, C.N_PERIODS)
    assert ((low >= 0) & (low <= 1)).all() and ((high >= 0) & (high <= 1)).all()
    assert (high.mean() > low.mean())
    assert (high >= low - 1e-12).mean() > 0.9           # bei gleicher Nachfrage meist besser
