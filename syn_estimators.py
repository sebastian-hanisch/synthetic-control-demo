"""Schätzer für den Effekt auf den behandelten Standort und Inferenz über Placebos.

y0: Servicegrad des behandelten Standorts je Periode, Form (R, N_PERIODS); yc: Kontrollen, Form (R, J, N_PERIODS). Perioden
0..T0-1 sind vorher, T0..N_PERIODS-1 nachher. Alle Schätzer liefern die mittlere Wirkung in der Nachperiode (Anteil, B minus
Gegenfaktisches)."""

import numpy as np
from scipy.optimize import nnls
from scipy.special import stdtr

import syn_constants as C

T0 = C.T0
T = C.N_PERIODS


def naive(y0):
    """Vorher-Nachher: Mittel der Nachperioden minus Mittel der Vorperioden des behandelten Standorts."""
    return y0[:, T0:].mean(axis=1) - y0[:, :T0].mean(axis=1)


def did(y0, yc):
    """Differenz-von-Differenzen aus den vier Zellmitteln."""
    return (y0[:, T0:].mean(axis=1) - y0[:, :T0].mean(axis=1)) - (yc[:, :, T0:].mean(axis=(1, 2)) - yc[:, :, :T0].mean(axis=(1, 2)))


def did_trend(y0, yc):
    """DiD mit Vortrend: Gerade durch die Differenzreihe (behandelt minus Mittel der Kontrollen) nur aus den Vorperioden,
    extrapoliert in die Nachperioden; Wirkung = mittlere Abweichung der Differenzreihe von dieser Geraden."""
    d = y0 - yc.mean(axis=1)
    t = np.arange(T, dtype=float)
    x = np.column_stack([np.ones(T0), t[:T0]])
    coef = np.linalg.lstsq(x, d[:, :T0].T, rcond=None)[0]
    fit = coef[0][:, None] + coef[1][:, None] * t[None, T0:]
    return (d[:, T0:] - fit).mean(axis=1)


def sc_weights(y_pre, c_pre, big=1e3):
    """Gewichte w >= 0 mit Summe 1, die ||y_pre - c_pre w|| minimieren (nnls mit Summenzeile). c_pre: (T0, J)."""
    j = c_pre.shape[1]
    a = np.vstack([c_pre, big * np.ones((1, j))])
    b = np.concatenate([y_pre, [big]])
    w, _ = nnls(a, b, maxiter=2000)
    return w


def sc(y0, yc):
    """Synthetische Kontrolle je Wiederholung. Rückgabe: tau (R,), Gewichte (R, J), RMSPE vorher (R,), RMSPE nachher (R,)."""
    r, j, _ = yc.shape
    tau = np.empty(r)
    weights = np.empty((r, j))
    rm_pre = np.empty(r)
    rm_post = np.empty(r)
    for i in range(r):
        w = sc_weights(y0[i, :T0], yc[i, :, :T0].T)
        gap = y0[i] - w @ yc[i]
        weights[i] = w
        tau[i] = gap[T0:].mean()
        rm_pre[i] = np.sqrt((gap[:T0] ** 2).mean())
        rm_post[i] = np.sqrt((gap[T0:] ** 2).mean())
    return tau, weights, rm_pre, rm_post


def estimate_all(y0, yc):
    """Alle vier Schätzer plus die SC-Zwischengrößen."""
    tau_sc, w, rm_pre, rm_post = sc(y0, yc)
    return {"naiv": naive(y0), "did": did(y0, yc), "did_trend": did_trend(y0, yc), "sc": tau_sc,
            "_w": w, "_rmspe_pre": rm_pre, "_rmspe_post": rm_post}


def placebo_taus(yc):
    """Placebo-Effekte: jede Kontrolle wird als behandelt behandelt, die übrigen sind ihre Kontrollen. yc (R, J, T).
    Rückgabe: dict Schätzer -> (R, J)."""
    r, j, _ = yc.shape
    out = {}
    pre_post = yc[:, :, T0:].mean(axis=2) - yc[:, :, :T0].mean(axis=2)                    # (R, J)
    out["naiv"] = pre_post
    others = (pre_post.sum(axis=1, keepdims=True) - pre_post) / (j - 1)
    out["did"] = pre_post - others
    d = yc - (yc.sum(axis=1, keepdims=True) - yc) / (j - 1)                                # (R, J, T)
    t = np.arange(T, dtype=float)
    x = np.column_stack([np.ones(T0), t[:T0]])
    pinv = np.linalg.pinv(x)
    coef = np.einsum("kt,rjt->krj", pinv, d[:, :, :T0])
    fit = coef[0][:, :, None] + coef[1][:, :, None] * t[None, None, T0:]
    out["did_trend"] = (d[:, :, T0:] - fit).mean(axis=2)
    tau = np.empty((r, j))
    for k in range(j):
        idx = [m for m in range(j) if m != k]
        tau[:, k] = sc(yc[:, k, :], yc[:, idx, :])[0]
    out["sc"] = tau
    return out


def placebo_pvalue(tau0, taup):
    """Rang-p-Wert: (1 + Zahl der Placebos mit |Effekt| >= |Effekt des Behandelten|) / (J + 1). Kleinster Wert 1 / (J + 1)."""
    return (1.0 + (np.abs(taup) >= np.abs(tau0)[:, None]).sum(axis=1)) / (taup.shape[1] + 1.0)


def twfe(y0, yc):
    """Zwei-Wege-Fixed-Effects mit einem behandelten Standort ab T0 (Frisch-Waugh, balanciertes Panel).
    Rückgabe: tau, se_iid, se_cluster (CR1, ein Cluster je Standort), Freiheitsgrade der iid-Rechnung (J + 1 Cluster -> J)."""
    y = np.concatenate([y0[:, None, :], yc], axis=1)
    r, s, t = y.shape
    d = np.zeros((s, t))
    d[0, T0:] = 1.0
    dt = d - d.mean(axis=1, keepdims=True) - d.mean(axis=0, keepdims=True) + d.mean()
    yt = y - y.mean(axis=2, keepdims=True) - y.mean(axis=1, keepdims=True) + y.mean(axis=(1, 2), keepdims=True)
    sdd = (dt**2).sum()
    tau = (yt * dt[None]).sum(axis=(1, 2)) / sdd
    e = yt - tau[:, None, None] * dt[None]
    n = s * t
    k = s + t
    df = n - k
    s2 = (e**2).sum(axis=(1, 2)) / df
    se_iid = np.sqrt(s2 / sdd)
    score = (dt[None] * e).sum(axis=2)
    v = (score**2).sum(axis=1) / sdd**2 * (s / (s - 1)) * ((n - 1) / (n - k))
    return tau, se_iid, np.sqrt(v), df


def t_pvalue(t, df):
    return 2.0 * stdtr(df, -np.abs(t))
