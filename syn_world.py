"""Das Vehikel: Standorte mit Order-up-to-Bestand, gemeinsamen Schocks und Nachfragetrends; Ergebnis ist der Servicegrad je Standort
und Periode (28 Tage).

Zeitablauf je Tag t und Standort s (alle Standorte und alle Wiederholungen gleichzeitig, numpy):
 1. Wareneingang: was am Ende von Tag t - LEAD bestellt wurde, kommt zum Netto-Bestand.
 2. Nachfrage D: negativ-binomial (Gamma-Poisson, Dispersion 5) mit Mittel = Standortgröße x Wochentag x gemeinsamer Wochenschock
    x Trend des Standorts. Bedient wird aus dem Bestand; die Fehlmenge ist Rückstand.
 3. Bestellung bis S = round(LEAD * mu + z * sd) über der Bestandsposition; das Zentrallager liefert unbegrenzt, die Standorte
    beeinflussen sich also nicht (keine Interferenz).
Aller Zufall kommt aus SplitMix64 (`syn_rng.uniform_block`); die negativ-binomiale Inversion läuft über `scipy.stats.nbinom.ppf`."""

import numpy as np
from scipy.special import ndtri
from scipy.stats import nbinom

import syn_constants as C
from syn_rng import uniform_block

DAYS = C.N_PERIODS * C.PERIOD_DAYS
TOTAL_DAYS = C.BURN_DAYS + DAYS
WEEKS = TOTAL_DAYS // 7 + 1


def trend_matrix(g):
    """Nachfragefaktor je Standort und Tag der Messtage: 1 + g * t / (DAYS - 1). g: (S,) Zuwachs über den ganzen Zeitraum."""
    g = np.asarray(g, dtype=float)
    return 1.0 + g[:, None] * (np.arange(DAYS)[None, :] / (DAYS - 1))


def scenario_trends(scenario, n_sites=C.S_ALL):
    """Zuwachs g je Standort für ein Szenario (der behandelte Standort ist C.TREATED)."""
    g = np.zeros(n_sites)
    if scenario == "gemeinsam":
        g[:] = C.TREND_G
    elif scenario == "eigen":
        g[C.TREATED] = C.TREND_G
    elif scenario == "verschieden":
        idx = np.arange(n_sites)
        g = -0.10 + 0.40 * ((idx * 11) % 31) / 30.0
        g[C.TREATED] = C.TREND_G
    return g


def scenario_shock(scenario):
    return 0.0 if scenario == "ruhig" else C.SHOCK_SD


def generate_demand(reps, sites, scenario, seed):
    """Nachfrage (reps, len(sites), BURN_DAYS + DAYS) als ganze Zahlen. Ein Block von C.CHUNK Wiederholungen hat seinen eigenen
    Zufallsstrom: das Ergebnis der ersten n Wiederholungen hängt nicht von der Gesamtzahl ab."""
    sites = np.asarray(sites, dtype=int)
    mu = C.site_means(C.S_ALL)[sites]
    dow = C.dow_factors()
    tr = trend_matrix(scenario_trends(scenario))[sites]                                 # (S, DAYS)
    tr = np.concatenate([np.repeat(tr[:, :1], C.BURN_DAYS, axis=1), tr], axis=1)       # (S, TOTAL)
    shock_sd = scenario_shock(scenario)
    dow_t = np.tile(dow, TOTAL_DAYS // 7 + 1)[:TOTAL_DAYS]
    out = np.empty((reps, len(sites), TOTAL_DAYS), dtype=np.int32)
    for c, start in enumerate(range(0, reps, C.CHUNK)):
        r = min(C.CHUNK, reps - start)
        block_seed = seed + 1_000_003 * c
        if shock_sd > 0:
            e = ndtri(np.clip(uniform_block(block_seed + 17, r * WEEKS).reshape(r, 1, WEEKS), 1e-12, 1 - 1e-12))
            shock = np.exp(shock_sd * e - 0.5 * shock_sd**2).repeat(7, axis=2)[:, :, :TOTAL_DAYS]
        else:
            shock = 1.0
        m = mu[None, :, None] * dow_t[None, None, :] * shock * tr[None, :, :]
        u = uniform_block(block_seed, r * len(sites) * TOTAL_DAYS).reshape(r, len(sites), TOTAL_DAYS)
        out[start:start + r] = nbinom.ppf(u, C.DISP, C.DISP / (C.DISP + m)).astype(np.int32)
    return out


def simulate_periods(demand, sites, z):
    """Servicegrad je Standort und Periode (reps, S, N_PERIODS). z: Sicherheitsfaktor, broadcastbar auf (reps, S, DAYS);
    die Einschwingtage nutzen z des ersten Messtages."""
    sites = np.asarray(sites, dtype=int)
    reps, s, total = demand.shape
    mu = C.site_means(C.S_ALL)[sites]
    sd = np.sqrt(C.LEAD * (mu + mu**2 / C.DISP))
    zb = np.broadcast_to(np.asarray(z, dtype=float), (reps, s, DAYS))
    zfull = np.concatenate([np.repeat(zb[:, :, :1], C.BURN_DAYS, axis=2), zb], axis=2)
    s_order = np.round(C.LEAD * mu[None, :, None] + zfull * sd[None, :, None])
    ni = np.round(C.LEAD * mu)[None, :].repeat(reps, axis=0) + np.round(zfull[:, :, 0] * sd[None, :])
    pipe = [np.zeros((reps, s)) for _ in range(C.LEAD)]
    short = np.empty((reps, s, DAYS))
    for t in range(total):
        ni = ni + pipe[0]
        d = demand[:, :, t].astype(float)
        served = np.minimum(np.maximum(ni, 0.0), d)
        ni = ni - d
        order = np.maximum(0.0, s_order[:, :, t] - (ni + sum(pipe[1:], start=np.zeros_like(ni))))
        pipe = pipe[1:] + [order]
        if t >= C.BURN_DAYS:
            short[:, :, t - C.BURN_DAYS] = d - served
    dem = demand[:, :, C.BURN_DAYS:].astype(float)
    p = C.PERIOD_DAYS
    return 1.0 - short.reshape(reps, s, C.N_PERIODS, p).sum(axis=3) / dem.reshape(reps, s, C.N_PERIODS, p).sum(axis=3)


def z_schedule(z_b):
    """Sicherheitsfaktor des behandelten Standorts je Messtag: Regel A vor T0, Regel B ab T0."""
    z = np.full(DAYS, C.Z_A)
    z[C.T0 * C.PERIOD_DAYS:] = z_b
    return z
