"""Auswertung: Schätzer und Placebo-Inferenz mit bekannter Wahrheit.

Je Wiederholung (eine simulierte Welt) gibt es einen behandelten Standort und J Kontrollen. Die Welt wird zweimal gerechnet
(gleiche Nachfrage): der behandelte Standort mit Regel A durchgehend (Nullwelt, Gegenfaktisches) und mit Regel B ab Periode T0
(beobachtete Welt). Die Wahrheit je Welt ist die mittlere Differenz der beiden in den Nachperioden."""

import numpy as np

import syn_constants as C
import syn_estimators as E
import syn_world as W


def seed_for(scenario, j, zb, tag=0, base=50_000_000):
    return base + 10_007 * C.SCENARIOS.index(scenario) + 131 * j + 17 * int(round(zb * 10)) + 13 * tag


def simulate_worlds(reps, scenario, j, zb, seed):
    """Perioden-Servicegrade: y0_obs (R, T) beobachtet, y0_nul (R, T) Gegenfaktisches, yc (R, J, T) Kontrollen."""
    sites = [C.TREATED] + list(C.CONTROLS[:j])
    demand = W.generate_demand(reps, sites, scenario, seed)
    y_a = W.simulate_periods(demand, sites, C.Z_A)
    y_b = W.simulate_periods(demand[:, :1, :], [C.TREATED], W.z_schedule(zb))
    return y_b[:, 0, :], y_a[:, 0, :], y_a[:, 1:, :]


P_LEVELS = (("0.05", 0.05), ("0.1", 0.10), ("0.2", 0.20))


def rate(x):
    return float(np.mean(x))


def summarize(reps, scenario, j, zb, tag=0, seed=None):
    """Kennzahlen je Schätzer in Prozentpunkten: Verzerrung, Streuung, RMSE (gegen die Wahrheit der jeweiligen Welt) und die
    Placebo-Tests (Anteil p <= 0,05 / 0,10 / 0,20 unter Null und bei Effekt); dazu iid-OLS und cluster-robust für DiD."""
    seed = seed_for(scenario, j, zb, tag) if seed is None else seed
    y_obs, y_nul, yc = simulate_worlds(reps, scenario, j, zb, seed)
    truth_rep = (y_obs - y_nul)[:, C.T0:].mean(axis=1)
    est_obs = E.estimate_all(y_obs, yc)
    est_nul = E.estimate_all(y_nul, yc)
    plc = E.placebo_taus(yc)
    rows = {}
    for name in C.ESTIMATORS:
        err = 100 * (est_obs[name] - truth_rep)
        p_nul = E.placebo_pvalue(est_nul[name], plc[name])
        p_obs = E.placebo_pvalue(est_obs[name], plc[name])
        rows[name] = {
            "bias": float(err.mean()), "bias_se": float(err.std(ddof=1) / np.sqrt(reps)),
            "sd": float(100 * est_obs[name].std(ddof=1)), "rmse": float(np.sqrt((err**2).mean())),
            "mean": float(100 * est_obs[name].mean()), "null_mean": float(100 * est_nul[name].mean()),
            "p_nul": {k: rate(p_nul <= a + 1e-12) for k, a in P_LEVELS},
            "p_obs": {k: rate(p_obs <= a + 1e-12) for k, a in P_LEVELS},
        }
    # Inferenz für DiD mit Standardfehlern (ein behandelter Standort)
    tau_o, se_iid_o, se_cr_o, df = E.twfe(y_obs, yc)
    tau_n, se_iid_n, se_cr_n, _ = E.twfe(y_nul, yc)
    from scipy.special import stdtrit
    t_iid = float(stdtrit(df, 0.975))
    t_cr = float(stdtrit(j, 0.975))
    inference = {
        "iid_null_reject": rate(np.abs(tau_n) > t_iid * se_iid_n),
        "cr_null_reject": rate(np.abs(tau_n) > t_cr * se_cr_n),
        "iid_obs_reject": rate(np.abs(tau_o) > t_iid * se_iid_o),
        "cr_obs_reject": rate(np.abs(tau_o) > t_cr * se_cr_o),
        "iid_coverage": rate(np.abs(tau_o - truth_rep) <= t_iid * se_iid_o),
        "cr_coverage": rate(np.abs(tau_o - truth_rep) <= t_cr * se_cr_o),
        "iid_width": float(200 * (t_iid * se_iid_o).mean()), "cr_width": float(200 * (t_cr * se_cr_o).mean()),
    }
    w = est_obs["_w"]
    sc_info = {"n_weights": float((w > 0.01).sum(axis=1).mean()), "max_weight": float(w.max(axis=1).mean()),
               "rmspe_pre": float(100 * est_obs["_rmspe_pre"].mean())}
    return {"scenario": scenario, "J": j, "zb": zb, "reps": reps, "truth": float(100 * truth_rep.mean()),
            "truth_sd": float(100 * truth_rep.std(ddof=1)), "rows": rows, "inference": inference, "sc": sc_info,
            "pre_level": float(100 * y_nul[:, :C.T0].mean()), "post_level": float(100 * y_nul[:, C.T0:].mean())}


def run_one(scenario, j, zb, seed):
    """Die eine Welt der Seitenleiste: Serien, Schätzungen, Synthetische Kontrolle, Placebos."""
    y_obs, y_nul, yc = simulate_worlds(1, scenario, j, zb, 60_000_000 + 7_919 * int(seed))
    est = E.estimate_all(y_obs, yc)
    plc = E.placebo_taus(yc)
    w = est["_w"][0]
    d = yc[0].mean(axis=0)
    # Gegenfaktisches nach DiD: Vorperiodenabstand zu den Kontrollen auf die Nachperioden übertragen
    shift = (y_obs[0, :C.T0] - d[:C.T0]).mean()
    return {"y0": y_obs[0], "y0_null": y_nul[0], "yc": yc[0], "w": w, "sc_line": w @ yc[0], "did_line": d + shift,
            "truth": float((y_obs[0] - y_nul[0])[C.T0:].mean()),
            "est": {k: float(est[k][0]) for k in C.ESTIMATORS},
            "placebo": {k: plc[k][0] for k in C.ESTIMATORS},
            "p": {k: float(E.placebo_pvalue(est[k], plc[k])[0]) for k in C.ESTIMATORS}}
