# Wenige Einheiten: Differenz-von-Differenzen und synthetische Kontrolle – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-synthetic-control-demo.streamlit.app/)** -->

Interaktive Demo zur **Wirkungsmessung mit einem behandelten Standort**. **Elftes Stück der Konzepte-Linie „Experimente und
Wirkungsmessung“** im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine
Learning). Das Vehikel ist die Bestandsregel der Vorgänger (tägliche Nachbestellung, Servicegrad), aber nicht mehr zufällig
zugeteilt: ein Standort bekommt die neue Regel, die übrigen nicht, und es gibt eine Zeitreihe mit Trends und gemeinsamen Schocks.

## Kernfrage

Wenn man nur einen Standort umstellen kann und es keine Zufallszuteilung gibt: was sagen Vorher-Nachher, Differenz-von-Differenzen
(DiD) und synthetische Kontrolle (SC) über die Wirkung, und wann tragen die Kontrollen nichts mehr?

## Modell und Methodik

- **Vehikel** (`syn_world.py`): bis zu 31 Standorte, Standortgrößen reihum, Wochentagsfaktor, negativ-binomiale Nachfrage
  (Gamma-Poisson), Bestellung bis zum Niveau `round(2 · mu + z · sd)`, Wiederbeschaffungszeit 2 Tage, Zentrallager unbegrenzt, die
  Standorte beeinflussen sich nicht. Beobachtet wird der **Servicegrad je 28-Tage-Periode**, 24 Perioden.
- **Behandlung:** Standort 3 (Größe 1,0) fährt ab Periode 17 Regel B (Faktor 1,2 bis 2,0 statt 1,0); 5, 12 oder 30 Kontrollen
  bleiben bei A. Die Wahrheit je Welt ist der Abstand zur Welt mit derselben Nachfrage ohne Umstellung (Nachperioden).
- **Fünf Szenarien:** ruhig; gemeinsame Wochenschocks (Streuung 0,15); gemeinsamer Trend (+20 % Nachfrage bei allen); eigener
  Trend (+20 % nur beim behandelten Standort); verschiedene Trends (Kontrollen −10 % bis +30 %, behandelter +20 %).
- **Vier Schätzer** (`syn_estimators.py`): Vorher-Nachher; DiD aus den vier Zellmitteln; DiD mit Vortrend (Gerade durch den
  Abstand der Vorperioden, in die Nachperioden verlängert); SC mit nichtnegativen Gewichten und Summe 1 (nnls).
- **Inferenz:** Placebo-Rangtest (jede Kontrolle wird als behandelt gerechnet; p = (1 + Zahl der Placebos mit mindestens so großem
  Effekt) / (J + 1)); dazu die gewöhnliche Regression mit iid-Fehlern und mit cluster-robusten Fehlern.
- **Simulationen:** 2 000 Welten je Zelle in `precomputed_sweeps.json` (23 Zellen), 400 Welten live in der App.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Prozentpunkte Servicegrad, Faktor 1,6 (wahre Wirkung 4,6 Punkte im ruhigen Szenario),
wenn nicht anders genannt. Verzerrung = mittlere Abweichung der Schätzung von der Wirkung der jeweiligen Welt; RMSE = typischer
Fehler in einer Welt.

| Frage | Befund |
|---|---|
| Hilft die Kontrolle, wenn nichts anderes passiert? | **Kaum.** Bei 12 Kontrollen RMSE Vorher-Nachher **2,31**, DiD **2,43**, SC **2,14**; bei 30 Kontrollen **2,33 / 2,36 / 2,14**. Alle ohne Verzerrung (unter 0,15). DiD fügt das Rauschen der Kontrollen hinzu, ohne etwas auszugleichen. |
| Und mit gemeinsamem Trend (+20 %)? | Wahre Wirkung **8,0**, Vorher-Nachher **−4,81** Punkte daneben (RMSE **6,10**); DiD **−0,06** (RMSE **3,24**), SC **−0,15** (**3,05**). |
| Und mit eigenem Trend des behandelten Standorts? | **Nichts rettet das.** Vorher-Nachher **−4,82**, DiD **−4,82** (RMSE **5,83**), SC **−7,82** (RMSE **8,39**: schlechter als gar keine Kontrolle). |
| Was hilft dann? | Der Vortrend entzerrt (Verzerrung **+0,02**), streut aber mit **5,45** Punkten bei wahrer Wirkung 8,0; RMSE **5,70**, kaum besser als DiD (5,83). |
| Bei verschiedenen Trends? | Vorher-Nachher **−4,86**, DiD **−2,55**, SC **−2,86** (12 Kontrollen); mit 30 Kontrollen DiD **−2,01**, SC **−2,12**. Der Vortrend ist unverzerrt (**−0,03**) bei RMSE **5,86**. |
| Wie gut passt die SC vorher? | Vorperioden-RMSPE **5,12 / 4,34 / 3,66** Punkte bei 5 / 12 / 30 Kontrollen; im Mittel **6,4** Kontrollen mit Gewicht über 1 %, größtes Gewicht **0,36** (30 Kontrollen). |
| Wie viele Welten findet der Placebo-Test? | Ruhig, 5 Kontrollen: unter **1/6 = 0,17** ist kein p-Wert möglich; auf 20 % findet die SC **74 %** (ohne Wirkung **20 %**). 12 Kontrollen, auf 10 %: SC **61 %**, DiD **49 %**; 30 Kontrollen, auf 5 %: SC **46 %**, DiD **34 %**, auf 10 % SC **72 %**. |
| Hält der Placebo-Test sein Niveau? | Bei 30 Kontrollen ohne Wirkung **3 bis 4 %** auf 5 % und **9 bis 10 %** auf 10 %. **Aber nicht bei eigenem Trend:** SC meldet auf 10 % in **85 %** der Welten (12 Kontrollen) einen Unterschied, auf 5 % bei 30 Kontrollen in **79 %**, DiD in **36 %**. |
| Wie wächst die Trennschärfe mit der Wirkung (30 Kontrollen, SC, p ≤ 0,1)? | Faktor 1,2 (Wirkung 2,0): **22 %**; 1,4: **46 %**; 1,6: **72 %**; 1,8: **83 %**; 2,0: **93 %**. |
| Cluster-robuste Standardfehler? | Mit **einem** behandelten Standort melden sie ohne Wirkung in **33 / 53 / 71 %** der Welten einen Unterschied (5 / 12 / 30 Kontrollen, versprochen 5 %), die Intervalle decken die Wahrheit in **67 / 47 / 30 %**: **schlechter mit mehr Kontrollen**. |
| Und die gewöhnliche Regression (iid)? | Im ruhigen Szenario **5,7 / 5,2 / 5,1 %** Fehlalarm. Mit eigenem Trend **40 / 43 / 46 %**. |
| Default-Seed 89 (gemeinsamer Trend, 12 Kontrollen) | Wahre Wirkung **7,27**; Vorher-Nachher **2,89**, DiD **7,07**, DiD mit Vortrend **7,78**, SC **7,66**; Placebo-p SC **0,077**, DiD **0,23**. |

## Befunde und Korrekturen gegenüber dem Plan

- **„Synthetische Kontrolle rettet bei nicht parallelen Trends“ stimmt nicht.** Bei eigenem Trend ist die SC schlechter als
  Vorher-Nachher (**−7,82** gegen **−4,82**): konvexe Gewichte können einen Trend, den keine Kontrolle hat, nicht nachbilden. In ruhigen Welten ist die SC der genaueste Schätzer (RMSE 2,14 gegen 2,31), aber
  nur um 7 %.
- **Der Placebo-Test hat bei eigenem Trend keine Gültigkeit mehr:** Die Placebos kennen den Trend des behandelten Standorts nicht.
  Der Test ist gültig, wenn die Standorte austauschbar sind, nicht darüber hinaus.
- **Mehr Kontrollen helfen bei der Trennschärfe, nicht bei der Verzerrung:** der Placebo-Test auf 5 % wird erst ab 19 Kontrollen
  überhaupt möglich (1/20), die Verzerrung bei verschiedenen Trends sinkt mit 30 Kontrollen aber nur von −2,55 auf −2,01.
- **Cluster-robuste Fehler sind hier kein Heilmittel.** Mit einem behandelten Cluster schätzen sie die Varianz aus einer einzigen
  Beobachtung; deshalb Fehlalarm bis 71 %. Die Demo zeigt, wie groß das Problem in diesem Vehikel ist.
- **Die gewöhnliche Regression ist hier nur deshalb brauchbar,** weil die Perioden kaum zusammenhängen (28-Tage-Fenster). Mit
  Serienkorrelation (Wochenwerte, kürzere Perioden) wäre sie es nicht.
- **Wochenwerte waren zu laut:** die Vorab-Messreihe mit Wochenwerten (Streuung 9,5 Punkte) fand die Wirkung kaum; deshalb
  28-Tage-Perioden und ein größerer Effekt (Faktor 1,6).

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ohne die Behandlung wären behandelter Standort und Kontrollen parallel gelaufen** | Eigener Trend des behandelten Standorts: DiD und SC liegen etwa so weit daneben wie Vorher-Nachher oder mehr. | (dieses Stück) |
| **Die Kontrollen sind von der Behandlung unberührt** | Weichen Kunden vom behandelten Standort zu Kontrollen aus, verfälscht das auch den Vergleich. | Interferenz und Switchback |
| **Genau ein behandelter Standort** | Cluster-robuste Standardfehler versagen; verlässlich sind der Placebo-Test und (hier) die gewöhnliche Regression. | (dieses Stück) |
| **Die SC kann den behandelten Standort nachbilden** | Auch in der Vorperiode bleibt ein Fehler von 3,7 bis 5,1 Punkten; ein schlechter Fit ist ein Warnzeichen. | (dieses Stück) |
| **Die Wirkung ist über die Nachperioden konstant** | Schwankt sie, mittelt der Schätzer sie zu einer Zahl. | (nicht Teil der Linie) |
| **Ein Test, eine Kennzahl** | Mehrere Kennzahlen oder Varianten vervielfachen die Tests. | Viele Kennzahlen: Mehrfachtests und Gewinnerfluch |

Die Zahlen gelten für dieses erfundene Vehikel; Trends und Schocks sind Modellannahmen.

## Bezug zu OR

Viele Betriebsumstellungen betreffen einen Standort, eine Linie oder einen Kunden und lassen sich nicht zufällig verteilen.
Dann hängt die Antwort an einer Annahme über die Kontrollen, die in den Daten nicht prüfbar ist; was man tun kann, ist die
Vorperiode ansehen (Vortrend) und ehrlich angeben, wie grob ein Placebo-p-Wert bei wenigen Kontrollen ist.

## Verwandte Demos im Portfolio

- [ab-test-demo](https://github.com/sebastian-hanisch/ab-test-demo): Vehikel und Welch-Test bei Zufallszuteilung.
- [switchback-demo](https://github.com/sebastian-hanisch/switchback-demo): Zuteilung, wenn Standorte einander beeinflussen.
- [forecast-interval-demo](https://github.com/sebastian-hanisch/forecast-interval-demo): Prognoseintervalle, strukturell nah am
  Gegenfaktischen über Extrapolation.
- [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo): die Base-Stock-Regel des Vehikels.

## Verifikation

- **Schätzer** (`tests/test_estimators.py`): von Hand an einer Mini-Instanz (vorher 0,90 / nachher 0,95, Kontrollen 0,92 / 0,94:
  Vorher-Nachher 0,05, DiD 0,03); DiD nimmt einen linearen Trend im Abstand mit, der Vortrend nicht; SC-Gewichte stellen einen
  exakten 0,5/0,5-Mix wieder her; Placebo-p von Hand; die Zwei-Wege-Fixed-Effects gegen die Regression mit Dummies.
- **Orakel** (`tests/test_oracle_synthetic_control.py`): Servicegrad gegen die exakte Formel der Order-up-to-Regel; SC-Gewichte
  gegen `scipy.optimize.minimize` (SLSQP) und gegen die KKT-Bedingungen; Placebo-p gegen direkte Aufzählung; DiD bei bekanntem
  Modell erwartungstreu bei parallelen Verläufen und um den Trendabstand verzerrt sonst.
- **Vehikel** (`tests/test_world.py`): Trend und Schock von Hand, gemeinsame Schocks (Korrelation), Nachfrage unabhängig von der
  Zahl der Welten.
- **Auswertung** (`tests/test_evaluation.py`): Wahrheit null ohne neue Regel; Verzerrung je Szenario; kleinster Placebo-p 1/(J+1).
- **Befunde:** jede Zahl dieses READMEs in `tests/test_claims.py`; zwei Zellen werden live nachgerechnet.
- **Presets, Oberfläche, Footer:** vollständig, Permalink eingerastet, AppTest-Rauchtest, Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: eine Welt, viele Welten, Szenarien, Trennschärfe, Grenzen |
| `syn_constants.py` | Vehikel-Konstanten, Szenarien, Schätzer, Presets |
| `syn_rng.py` | SplitMix64 |
| `syn_world.py` | Nachfrage mit Schock und Trend, Tagesablauf, Servicegrad je Periode |
| `syn_estimators.py` | Vorher-Nachher, DiD, DiD mit Vortrend, SC, Placebos, Zwei-Wege-Regression |
| `syn_evaluation.py` | Welten, Wahrheit, Zusammenfassungen, eine Welt |
| `syn_presets.py`, `syn_visualization.py` | Permalink, Presets; Plotly-Abbildungen (Achsen gesperrt) |
| `generate_precomputed.py`, `precomputed_sweeps.json` | Sweeps (23 Zellen à 2 000 Welten) |
| `tests/` | Vehikel, Schätzer, Orakel, Auswertung, Presets, Befunde, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Mehrere behandelte Standorte, gestaffelter Beginn** und moderne DiD-Schätzer dafür: nicht gebaut.
- **Konforme Intervalle** und andere Inferenz für SC als der Placebo-Rangtest: nicht gebaut.
- **SC mit Achsenabschnitt** (zentrierte Reihen) und mit Kovariaten: nicht gebaut.
- **Serienkorrelation** in den Perioden: nicht abgebildet.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy (nnls, Verteilungsfunktionen).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
