"""Wenige Einheiten: Differenz-von-Differenzen und synthetische Kontrolle - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes Stück der Konzepte-Linie "Experimente und Wirkungsmessung": Ein einzelner Standort bekommt eine neue Bestandsregel, die
übrigen nicht. Wie viel lässt sich aus Vorher-Nachher, Differenz-von-Differenzen und synthetischer Kontrolle über die Wirkung
sagen, und wann tragen die Kontrollen nichts mehr? Bekannte Wirkung, vier Schätzer, Placebo-Inferenz. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

import syn_constants as C
from syn_evaluation import run_one, summarize
from syn_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed,
                         sync_query_params)
from syn_visualization import build_bias_by_scenario, build_estimates, build_placebo, build_power, build_series

st.set_page_config(page_title="Synthetische Kontrolle – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _sweeps():
    data = json.loads((Path(__file__).resolve().parent / "precomputed_sweeps.json").read_text(encoding="utf-8"))
    return data["cells"]


@st.cache_data(show_spinner=False)
def _one(scenario, j, zb, seed):
    return run_one(scenario, j, zb, seed)


@st.cache_data(show_spinner=False)
def _many(scenario, j, zb):
    return summarize(C.WORLDS_LIVE, scenario, j, zb)


def _cell(sweep, scenario, j, zb=C.DEFAULT_ZB):
    for c in _sweeps():
        if c["sweep"] == sweep and c["scenario"] == scenario and c["J"] == j and abs(c["zb"] - zb) < 1e-9:
            return c
    raise KeyError((sweep, scenario, j, zb))


st.title("🧮 Wenige Einheiten: Differenz-von-Differenzen und synthetische Kontrolle")
st.markdown(
    """
Ein einzelner Standort bekommt eine neue Bestandsregel, **die übrigen nicht**. Ein A/B-Test ist das nicht: es gibt einen
behandelten Standort und eine Handvoll Kontrollen. Was bleibt, ist der **Vergleich über die Zeit**: *Vorher-Nachher* misst
alles mit, was sich sonst geändert hat; *Differenz-von-Differenzen* (DiD) rechnet aus, was sich bei den Kontrollen im selben
Zeitraum geändert hat; die *synthetische Kontrolle* (SC) baut aus den Kontrollen einen gewichteten Doppelgänger des behandelten
Standorts. Alle drei beruhen auf **Annahmen, die man nicht prüfen kann**, und auf einer Unsicherheit, die mit nur einem
behandelten Standort grob bleibt.
"""
)
st.caption(
    "Elftes Stück der Linie „Experimente und Wirkungsmessung“, aufbauend auf "
    "[switchback-demo](https://sebastianhanisch-switchback-demo.streamlit.app/) (Zuteilung bei Interferenz) und "
    "[ab-test-demo](https://sebastianhanisch-ab-test-demo.streamlit.app/) (gleiches Vehikel: Servicegrad aus der Bestandsregel). "
    "Neu: eine **Zeitreihe** je Standort mit Trend und gemeinsamen Schocks."
)

with st.expander("So ist das Experiment aufgebaut", expanded=True):
    st.markdown(
        """
- **Ein behandelter Standort, J Kontrollen** (5, 12 oder 30), 24 Perioden zu je 28 Tagen. In den ersten 16 Perioden fahren alle
  Regel A (Sicherheitsfaktor 1,0); ab Periode 17 fährt der behandelte Standort Regel B, die Kontrollen bleiben bei A.
- **Wahrheit:** dieselbe Nachfrage, einmal mit A und einmal mit B beim behandelten Standort. Die wahre Wirkung ist der Abstand in
  den Nachperioden (Servicegrad, Prozentpunkte). Die Standorte beeinflussen sich nicht.
- **Vier Schätzer:** *Vorher-Nachher* (nur der behandelte Standort), *DiD* (Vorher-Nachher minus Vorher-Nachher der
  Kontrollen), *DiD mit Vortrend* (zusätzlich eine Gerade durch den Abstand in den Vorperioden, in die Nachperioden
  verlängert) und *synthetische Kontrolle* (nichtnegative Gewichte mit Summe 1, die den behandelten Standort in den
  Vorperioden nachbilden).
- **Inferenz:** der Placebo-Test behandelt jede Kontrolle, als wäre sie behandelt, und zählt, wie oft ein Placebo einen so
  großen Effekt zeigt. Mit J Kontrollen ist der kleinste mögliche p-Wert 1/(J+1).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{i}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])
st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Wahrheit**")
    scen_index = st.radio("Was ändert sich sonst noch?", options=list(range(len(C.SCENARIOS))), key="scen_radio",
                          format_func=lambda i: C.SCENARIO_LABELS[C.SCENARIOS[i]],
                          help="Gemeinsame Schocks und Trends wirken auf alle Standorte; ein eigener Trend nur auf den behandelten.")
    zb = st.slider("Sicherheitsfaktor der neuen Regel B", *bounds("zb_slider"), step=C.ZB_STEP, key="zb_slider", format="%.1f",
                   help="Regel A fährt 1,0. Je größer B, desto größer die wahre Wirkung.")
    st.markdown("**Experiment**")
    j_index = st.radio("Kontrollstandorte", options=list(range(len(C.J_CHOICES))), key="j_radio",
                       format_func=lambda i: f"{C.J_CHOICES[i]} Kontrollen",
                       help="Aus wie vielen Standorten Vergleichsreihen und Placebos entstehen.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1,
                           key="seed_input", help="Bestimmt die eine Welt unten.")
    st.button("🎲 Neue Welt würfeln", on_click=randomize_seed)

scen_index, j_index, seed = int(scen_index), int(j_index), int(seed)
zb = round(float(zb), 1)
scenario, j = C.SCENARIOS[scen_index], C.J_CHOICES[j_index]
sync_query_params({"scen_radio": scen_index, "j_radio": j_index, "zb_slider": zb, "seed_input": seed})

# --- Eine Welt ------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🧪 Eine Welt")
one = _one(scenario, j, zb, seed)
tag = f"{scen_index}_{j}_{zb}_{seed}"
st.caption(f"{C.SCENARIO_LABELS[scenario]}; {j} Kontrollen; Regel B mit Faktor {C.fmt_num(zb)}. Wahre Wirkung in dieser Welt: "
           f"**{C.fmt_pp(one['truth'])}**.")
st.plotly_chart(build_series(one), width="stretch", key=f"series_{tag}")
st.plotly_chart(build_estimates(one["est"], one["truth"]), width="stretch", key=f"est_{tag}")
rows_md = ["| Schätzer | Geschätzte Wirkung | Abstand zur Wahrheit | Placebo-p |", "|---|---|---|---|"]
for k in C.ESTIMATORS:
    rows_md.append(f"| {C.ESTIMATOR_LABELS[k]} | {C.fmt_pp(one['est'][k])} | {C.fmt_pp(one['est'][k] - one['truth'])} | "
               f"{C.fmt_num(one['p'][k], 2)} |")
st.markdown("\n".join(rows_md))
st.plotly_chart(build_placebo(one["placebo"], one["est"], one["p"], j), width="stretch", key=f"placebo_{tag}")
st.caption(f"Mit {j} Placebo-Kontrollen kann der Placebo-p-Wert nicht kleiner werden als 1/{j + 1} = "
           f"{C.fmt_num(1 / (j + 1), 3)}.")

# --- Viele Welten ---------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📐 Viele Welten: welcher Schätzer trifft?")
with st.spinner(f"Rechne {C.WORLDS_LIVE} Welten …"):
    many = _many(scenario, j, zb)
rows = many["rows"]
tab = ["| Schätzer | Verzerrung | Streuung | Typischer Fehler (RMSE) | Placebo-p ≤ 0,2 (ohne Wirkung / mit) | p ≤ 0,1 | p ≤ 0,05 |",
       "|---|---|---|---|---|---|---|"]
for k in C.ESTIMATORS:
    r = rows[k]
    tab.append(
        f"| {C.ESTIMATOR_SHORT[k]} | {C.fmt_num(r['bias'], 2)} | {C.fmt_num(r['sd'], 2)} | {C.fmt_num(r['rmse'], 2)} | "
        f"{C.fmt_pct(r['p_nul']['0.2'], 0)} / {C.fmt_pct(r['p_obs']['0.2'], 0)} | "
        f"{C.fmt_pct(r['p_nul']['0.1'], 0)} / {C.fmt_pct(r['p_obs']['0.1'], 0)} | "
        f"{C.fmt_pct(r['p_nul']['0.05'], 0)} / {C.fmt_pct(r['p_obs']['0.05'], 0)} |"
    )
st.markdown("\n".join(tab))
st.caption(f"Alle Zahlen in Prozentpunkten Servicegrad; Wahrheit im Mittel {C.fmt_num(many['truth'], 2)} (Streuung der wahren Wirkung "
           f"{C.fmt_num(many['truth_sd'], 2)}). Verzerrung = mittlere Abweichung von der Wirkung der jeweiligen Welt.")
inf = many["inference"]
st.markdown(
    f"**Standardfehler-Tests für DiD** (Test auf 5 %): ohne Wirkung meldet die übliche Regression (iid-Fehler) in "
    f"{C.fmt_pct(inf['iid_null_reject'])} der Welten einen Unterschied, die cluster-robuste in "
    f"**{C.fmt_pct(inf['cr_null_reject'], 0)}**; die Intervalle decken die Wahrheit in {C.fmt_pct(inf['iid_coverage'], 0)} "
    f"bzw. {C.fmt_pct(inf['cr_coverage'], 0)}."
)
best = min(C.ESTIMATORS, key=lambda k: rows[k]["rmse"])
if scenario == "eigen":
    st.error(
        f"**Der behandelte Standort geht eigene Wege, und keine Auswertung rettet das.** DiD liegt im Mittel "
        f"{C.fmt_num(rows['did']['bias'], 1)} Punkte daneben, die synthetische Kontrolle {C.fmt_num(rows['sc']['bias'], 1)}; der "
        f"Vortrend entzerrt ({C.fmt_num(rows['did_trend']['bias'], 1)}), streut aber um {C.fmt_num(rows['did_trend']['sd'], 1)} "
        f"Punkte bei einer wahren Wirkung von {C.fmt_num(many['truth'], 1)}. Die Annahme „ohne Behandlung wären beide parallel "
        "gelaufen“ ist verletzt, und die Daten allein zeigen das nicht zuverlässig."
    )
elif scenario in ("gemeinsam", "verschieden"):
    st.info(
        f"Vorher-Nachher liegt im Mittel {C.fmt_num(rows['naiv']['bias'], 1)} Punkte daneben, weil der gemeinsame Trend mitgemessen "
        f"wird; DiD {C.fmt_num(rows['did']['bias'], 1)}, synthetische Kontrolle {C.fmt_num(rows['sc']['bias'], 1)}. Kleinster "
        f"typischer Fehler: {C.ESTIMATOR_SHORT[best]} ({C.fmt_num(rows[best]['rmse'], 1)} Punkte)."
    )
else:
    st.info(
        f"Ohne gemeinsame Trends hilft die Kontrolle kaum: Vorher-Nachher {C.fmt_num(rows['naiv']['rmse'], 2)}, DiD "
        f"{C.fmt_num(rows['did']['rmse'], 2)}, synthetische Kontrolle {C.fmt_num(rows['sc']['rmse'], 2)} Punkte typischer Fehler. "
        "DiD braucht Kontrollen für Dinge, die es hier nicht gibt, und fügt ihr eigenes Rauschen hinzu."
    )

# --- Szenarien ------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Alle Szenarien nebeneinander")
scen_cells = [_cell("scenario", s, j)["rows"] for s in C.SCENARIOS]
labels_short = ["Ruhig", "Schocks", "Gemeinsamer Trend", "Eigener Trend", "Verschiedene Trends"]
st.plotly_chart(build_bias_by_scenario(labels_short, scen_cells, "bias",
                                       f"Verzerrung je Szenario ({j} Kontrollen, Faktor 1,6)", "Prozentpunkte"),
                width="stretch", key=f"bias_{j}")
st.plotly_chart(build_bias_by_scenario(labels_short, scen_cells, "rmse",
                                       f"Typischer Fehler (RMSE) je Szenario ({j} Kontrollen, Faktor 1,6)", "Prozentpunkte"),
                width="stretch", key=f"rmse_{j}")
st.caption("Die Zahlen dieses Abschnitts sind vorgerechnet (je 2 000 Welten, Faktor 1,6) und gehören zur gewählten Zahl "
           "der Kontrollen; Szenario und Faktor der Seitenleiste wirken nicht auf diese Abbildungen.")

# --- Trennschärfe ---------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 Wie groß muss die Wirkung sein, damit der Placebo-Test sie findet?")
zbs = (1.2, 1.4, 1.6, 1.8, 2.0)
series = {}
for jj in (12, 30):
    cells = [_cell("effect", "ruhig", jj, z) if abs(z - 1.6) > 1e-9 else _cell("scenario", "ruhig", jj, 1.6) for z in zbs]
    series[f"Synthetische Kontrolle, {jj} Kontrollen (p ≤ 0,1)"] = (
        [c["rows"]["sc"]["p_obs"]["0.1"] for c in cells], C.ESTIMATOR_COLORS["sc"], "solid" if jj == 30 else "dot")
    series[f"DiD, {jj} Kontrollen (p ≤ 0,1)"] = (
        [c["rows"]["did"]["p_obs"]["0.1"] for c in cells], C.ESTIMATOR_COLORS["did"], "solid" if jj == 30 else "dot")
st.plotly_chart(build_power(list(zbs), series, "Anteil der ruhigen Welten, in denen der Placebo-Test auf 10 % die Wirkung findet",
                            "Sicherheitsfaktor der neuen Regel B"), width="stretch", key="power_chart")
c12, c30 = _cell("scenario", "ruhig", 12), _cell("scenario", "ruhig", 30)
st.info(
    f"Bei Faktor 1,6 (wahre Wirkung {C.fmt_num(c12['truth'], 1)} Punkte) findet die synthetische Kontrolle sie mit 12 Kontrollen "
    f"in {C.fmt_pct(c12['rows']['sc']['p_obs']['0.1'], 0)} der Welten, mit 30 Kontrollen in "
    f"{C.fmt_pct(c30['rows']['sc']['p_obs']['0.1'], 0)}; ein Placebo-Test auf 5 % ist mit 12 Kontrollen gar nicht möglich "
    f"(kleinster p-Wert {C.fmt_num(1 / 13, 3)})."
)

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ohne die Behandlung wären behandelter Standort und Kontrollen parallel gelaufen** | Eigener Trend des behandelten Standorts: DiD und synthetische Kontrolle liegen etwa so weit daneben wie Vorher-Nachher oder mehr. Prüfbar ist das nur an den Vorperioden. | (dieses Stück) |
| **Die Kontrollen sind von der Behandlung unberührt** | Weichen Kunden vom behandelten Standort zu Kontrollen aus, verfälscht das auch den Vergleich. | Interferenz und Switchback |
| **Genau ein behandelter Standort** | Cluster-robuste Standardfehler unterschätzen die Unsicherheit stark (ohne Wirkung melden sie in 33 / 53 / 71 % der Welten einen Unterschied bei 5 / 12 / 30 Kontrollen); verlässlich sind der Placebo-Test und, weil die Perioden hier kaum zusammenhängen, die gewöhnliche Regression. | (dieses Stück) |
| **Die synthetische Kontrolle kann den behandelten Standort nachbilden** | Hier bleibt auch in der Vorperiode ein Fehler von 3,7 bis 5,1 Punkten (RMSPE), weil die Periodenwerte selbst rauschen; ein schlechter Fit ist ein Warnzeichen, kein Beweis. | (dieses Stück) |
| **Die Wirkung ist über die Nachperioden konstant** | Schwankt sie, mittelt der Schätzer sie zu einer Zahl. | (nicht Teil der Linie) |
| **Ein Test, eine Kennzahl** | Mehrere Kennzahlen oder Varianten vervielfachen die Tests. | Viele Kennzahlen: Mehrfachtests und Gewinnerfluch |
"""
)
st.caption(
    "Die Zahlen stammen aus dem erfundenen Vehikel dieses Stücks (24 Perioden, ein behandelter Standort, unabhängige Standorte, "
    "Zentrallager unbegrenzt). Trends und Schocks sind Modellannahmen, kein Betriebsbefund."
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**DiD.** Mit Servicegrad $y_{it}$, behandeltem Standort $i = 0$ und Nachperioden $t \ge T_0$ ist
$\hat\tau_{\mathrm{DiD}} = (\bar y_{0,\mathrm{nach}} - \bar y_{0,\mathrm{vor}}) - (\bar y_{c,\mathrm{nach}} - \bar y_{c,\mathrm{vor}})$,
mit $\bar y_c$ als Mittel über alle Kontrollen. Unverzerrt, wenn der Abstand zwischen behandeltem Standort und Kontrollen ohne
Behandlung konstant geblieben wäre (parallele Verläufe).

**DiD mit Vortrend.** Eine Gerade $a + b\,t$ wird nur an den Abstand $d_t = y_{0t} - \bar y_{ct}$ der Vorperioden angepasst und in
die Nachperioden verlängert; $\hat\tau = \frac{1}{T - T_0} \sum_{t \ge T_0} (d_t - a - b\,t)$. Entzerrt einen eigenen linearen
Trend, verstärkt aber das Rauschen, weil die Steigung aus wenigen Vorperioden kommt.

**Synthetische Kontrolle.** Gewichte $w \ge 0$, $\sum_j w_j = 1$, die $\sum_{t < T_0} (y_{0t} - \sum_j w_j y_{jt})^2$ minimieren;
$\hat\tau_{\mathrm{SC}} = \frac{1}{T - T_0} \sum_{t \ge T_0} (y_{0t} - \sum_j w_j y_{jt})$. Konvexe Gewichte verbieten
Extrapolation: liegt der behandelte Standort außerhalb der Kontrollen, bleibt eine Lücke.

**Placebo-Test.** Jede Kontrolle wird als behandelt gerechnet (mit den übrigen als Kontrollen); der Rang-p-Wert ist
$\frac{1 + \#\{j: |\hat\tau_j| \ge |\hat\tau_0|\}}{J + 1}$, mit kleinstem Wert $1/(J+1)$. Er braucht keine Annahme über
die Streuung der Fehler, nur austauschbare Standorte.

**Cluster-robuste Standardfehler** setzen viele behandelte Cluster voraus; bei einem behandelten Standort unterschätzen sie die
Unsicherheit stark (siehe Befunde).

Implementiert in `syn_world.py` (Vehikel mit Trends), `syn_estimators.py` (Schätzer, Placebos) und `syn_evaluation.py`.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Experimente und Wirkungsmessung: A/B bis Logdaten](https://sebastianhanisch.net/konzepte-experimente-wirkungsmessung.html)."
)
