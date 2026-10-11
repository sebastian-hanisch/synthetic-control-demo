"""Konstanten: Vehikel, Regler-Grenzen, Farben, Presets. Das Vehikel (Standorte mit Order-up-to-Bestand) ist das der Linie."""

import numpy as np

# Vehikel: Standorte mit täglicher Nachbestellung (Order-up-to), Wiederbeschaffungszeit 2 Tage, Zentrallager unbegrenzt.
BASE_MEAN = 20.0          # mittlere Tagesnachfrage der Standortgröße 1
SIZES = (0.6, 0.8, 0.9, 1.0, 1.0, 1.1, 1.3, 1.5)          # Standortgrößen relativ zur Basis (reihum)
DOW = (1.15, 1.10, 1.00, 1.00, 1.05, 0.90, 0.80)          # Wochentagsfaktor (Montag zuerst), wird auf Mittel 1 normiert
DISP = 5                  # Dispersion der negativ-binomialen Tagesnachfrage (Gamma-Poisson)
LEAD = 2                  # Wiederbeschaffungszeit in Tagen
BURN_DAYS = 28            # Einschwingtage vor der ersten Periode
Z_A = 1.0                 # Sicherheitsfaktor der bestehenden Regel A

# Stück 11: wenige Einheiten
PERIOD_DAYS = 28          # eine Periode = ein Standort-Fenster (vier Wochen)
N_PERIODS = 24            # beobachtete Perioden je Standort
T0 = 16                   # Perioden 0..15 vorher, 16..23 nachher
TREATED = 3               # der behandelte Standort (Größe 1,0)
S_ALL = 31                # simulierte Standorte: der behandelte und bis zu 30 Kontrollen
CONTROLS = tuple(i for i in range(S_ALL) if i != TREATED)
ZB_MIN, ZB_MAX, DEFAULT_ZB, ZB_STEP = 1.2, 2.0, 1.6, 0.2  # Sicherheitsfaktor der neuen Regel B beim behandelten Standort
J_CHOICES = (5, 12, 30)   # Zahl der Kontrollstandorte
DEFAULT_J_INDEX = 1       # Index in J_CHOICES: 12 Kontrollen
SHOCK_SD = 0.15           # gemeinsame Wochenschocks der Nachfrage (lognormal, alle Standorte)
TREND_G = 0.20            # Nachfragezuwachs über den ganzen Zeitraum bei Trend-Szenarien (+20 %)
SEED_MAX, DEFAULT_SEED = 999999, 89
WORLDS_LIVE = 400         # simulierte Wiederholungen in der Live-Auswertung
CHUNK = 100               # Wiederholungen je Erzeugungsblock (Zufallsstrom je Block: Ergebnis hängt nicht von der Gesamtzahl ab)

SCENARIOS = ("ruhig", "schock", "gemeinsam", "eigen", "verschieden")
SCENARIO_LABELS = {
    "ruhig": "Ruhig: kein Schock, kein Trend",
    "schock": "Gemeinsame Wochenschocks",
    "gemeinsam": "Gemeinsamer Trend (+20 % Nachfrage bei allen)",
    "eigen": "Eigener Trend: nur der behandelte Standort wächst",
    "verschieden": "Verschiedene Trends (Kontrollen −10 % bis +30 %)",
}
ESTIMATORS = ("naiv", "did", "did_trend", "sc")
ESTIMATOR_LABELS = {
    "naiv": "Vorher-Nachher (nur der behandelte Standort)",
    "did": "Differenz-von-Differenzen (DiD)",
    "did_trend": "DiD mit Vortrend des Unterschieds",
    "sc": "Synthetische Kontrolle (SC)",
}
ESTIMATOR_SHORT = {"naiv": "Vorher-Nachher", "did": "DiD", "did_trend": "DiD + Vortrend", "sc": "Synthetische Kontrolle"}
ESTIMATOR_COLORS = {"naiv": "#c0392b", "did": "#4c78a8", "did_trend": "#8c564b", "sc": "#2F6B65"}

COLOR_TREATED = "#d95f02"
COLOR_CONTROL = "rgba(120,120,120,0.45)"
COLOR_TRUTH = "#d95f02"

PRESETS = {
    "Gemeinsamer Trend (Voreinstellung)": {"scen": 2, "j": DEFAULT_J_INDEX, "zb": DEFAULT_ZB, "seed": DEFAULT_SEED},
    "Ruhig, wenige Kontrollen": {"scen": 0, "j": 0, "zb": DEFAULT_ZB, "seed": DEFAULT_SEED},
    "Eigener Trend": {"scen": 3, "j": DEFAULT_J_INDEX, "zb": DEFAULT_ZB, "seed": DEFAULT_SEED},
    "Viele Kontrollen, kleiner Effekt": {"scen": 1, "j": 2, "zb": 1.2, "seed": DEFAULT_SEED},
}
PRESET_ORDER = tuple(PRESETS)
PRESET_HELP = {
    "Gemeinsamer Trend (Voreinstellung)": "Die Nachfrage wächst bei allen Standorten um 20 %: Vorher-Nachher misst den Trend mit, DiD rechnet ihn heraus.",
    "Ruhig, wenige Kontrollen": "Kein Schock, kein Trend, fünf Kontrollen: der günstigste Fall, und doch gibt es nur grobe Aussagen über die Sicherheit.",
    "Eigener Trend": "Nur der behandelte Standort wächst um 20 %: die Annahme paralleler Verläufe ist verletzt, und die Kontrollen können nichts ausgleichen.",
    "Viele Kontrollen, kleiner Effekt": "30 Kontrollen, Wirkung klein: wie viel Sicherheit gibt der Placebo-Test jetzt?",
}


def site_means(n_sites=S_ALL):
    sizes = np.array(SIZES, dtype=float)
    return BASE_MEAN * np.tile(sizes, int(np.ceil(n_sites / len(sizes))))[:n_sites]


def dow_factors():
    d = np.array(DOW, dtype=float)
    return d / d.mean()


def fmt_pct(x, digits=1):
    """Anteil (0..1) als Prozentzahl mit Dezimalkomma, z. B. 0,934 -> '93,4 %'."""
    return f"{100 * x:.{digits}f}".replace(".", ",") + " %"


def fmt_pp(x, digits=2):
    """Differenz in Prozentpunkten (aus Anteilen 0..1) mit Vorzeichen und Dezimalkomma."""
    return f"{100 * x:+.{digits}f}".replace(".", ",") + " Prozentpunkte"


def fmt_pkt(x, digits=1):
    """Kurzform für enge Kacheln: '+4,0 Pkt.' (Differenz in Prozentpunkten, aus Anteilen 0..1)."""
    return f"{100 * x:+.{digits}f}".replace(".", ",") + " Pkt."


def fmt_num(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_int(x):
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(x):,}".replace(",", " ")
