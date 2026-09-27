# 🤖 Q-Learning

**[→ Demo live ausprobieren](https://sebastianhanisch-q-learning-demo.streamlit.app/)**

Drittes Stück der **Reinforcement-Learning-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Derselbe Lagerroboter wie bei [Value Iteration und Policy Iteration](https://github.com/sebastian-hanisch/value-iteration-demo) (Stück 2) – Raster, Klippe, Packstation, Rutsch-Wahrscheinlichkeit. Diesmal kennt der Roboter das Modell **nicht**: **Q-Learning** (Watkins & Dayan 1992) lernt die optimale Policy allein aus wiederholtem Ausprobieren (Episoden), ohne je die Bellman-Gleichung direkt auf einem Modell auszuwerten.

## Kernfrage

**Konvergiert Q-Learning auf dieselbe Policy wie Value Iteration – und zu welchem Preis?** Wie viele Umgebungsschritte braucht das Ausprobieren gegenüber den Bellman-Backups, die Value Iteration auf demselben (hier für die Gegenprobe bekannten) Modell braucht? Und was schiefgeht, wenn zu wenig trainiert, zu wenig erkundet oder zu grob gelernt wird?

## Modell

- **Vehikel** (`ql_grid.py`): dasselbe Raster wie in `value-iteration-demo` (Cliff-Walking-Vorlage, Sutton & Barto 2018, Beispiel 6.6). Neu hier: der Agent sieht nur einzelne Übergänge $(s, a, r, s')$ aus `step` – nie die Wahrscheinlichkeiten oder erwarteten Belohnungen selbst. Das Modell (`build_model`) existiert im Code nur noch für die Referenzlösung (Value Iteration, zur Gegenprobe), niemals für den lernenden Agenten.
- **TD-Update** (Watkins & Dayan 1992): $Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big)$, initialisiert mit $Q_0 \equiv 0$.
- **Off-Policy:** das Verhalten während des Lernens ist **epsilon-gierig**, das Lernziel ist immer die gierige Aktion – unabhängig davon, was tatsächlich getan wird.
- **GLIE** (greedy in the limit with infinite exploration): $\varepsilon_e = \max(\varepsilon_{\min}, \varepsilon_0 / (1 + \text{decay} \cdot e))$ fällt gegen null, aber nie ganz auf null.

## Methodik

Bewertet wird nicht "trifft die gelernte Policy exakt dieselbe Aktion wie Value Iteration in jeder Zelle" (irrelevant für Zellen, die die Policy nie besucht), sondern ihr tatsächlicher **Wert**: $V^*(\text{Start}) - V_{\pi_{\text{gelernt}}}(\text{Start})$, wobei $V_{\pi_{\text{gelernt}}}$ die gelernte Policy exakt unter dem bekannten Modell auswertet (Policy Evaluation). Drei Experimente: Wert-Abstand über die Trainingsdauer (gegen Value Iterations Bellman-Backups), Wirkung des Epsilon-Zerfalls (GLIE gegen konstant explorativ), Wirkung der Lernrate α. Alle Experimente über mehrere Seeds (Q-Learning ist stochastisch, anders als die exakte Referenz).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (4×8-Raster, Rutschen 0,10, α=0,10, Epsilon-Zerfall 0,005, 1500 Episoden, Seed 0) | V*(Start) exakt −9,23; gelernte Policy erreicht −11,62 (Abstand 2,40) – die Policy bleibt am Start bei "gegen die Westwand laufen" hängen statt aktiv nach Norden von der Klippe wegzugehen, weil beides in der Q-Schätzung fast gleich billig aussieht. 96,2 % der übrigen Zellen stimmen mit der optimalen Policy überein. 52.917 Umgebungsschritte gegenüber 1.408 Bellman-Backups von Value Iteration auf demselben Modell (37×). | `test_standard_case` |
| **Löst mehr Training das zuverlässig?** (100 bis 4000 Episoden, 20 Seeds je Stufe) | Nein: der Anteil nahezu optimaler Läufe bleibt über die ganze Spanne bei etwa der Hälfte (50 % bei 100 Episoden, 55 % bei 4000 Episoden) – die "Westwand"-Gewohnheit ist genauso billig wie die echte Route, mehr Training allein löst den Beinah-Gleichstand nicht auf, weil Epsilon inzwischen kaum noch erkundet. Selbst bei 4000 Episoden braucht Q-Learning 107.147 Umgebungsschritte im Mittel – das 76-fache von Value Iterations 1.408 Bellman-Backups. | `test_episodes_experiment` |
| **Wie wichtig ist GLIE (Epsilon-Zerfall)?** (0 bis 0,02, 800 Episoden, 20 Seeds je Stufe) | Ohne Zerfall (Epsilon bleibt bei 1,0, also die ganze Trainingszeit rein zufälliges Verhalten) erreichen nur 30 % der Läufe eine nahezu optimale Policy – **obwohl** dieser Fall im Mittel 7,6× so viele Umgebungsschritte verbraucht wie die beste gemessene Zerfallsrate (0,005: 60 % nahezu optimal). Mehr Erfahrung ersetzt GLIE nicht. | `test_decay_experiment` |
| **Wie stark wirkt die Lernrate α?** (0,05 bis 0,50, 800 Episoden, 20 Seeds je Stufe) | Kleine Lernraten liefern verlässlicher eine nahezu optimale Policy (α=0,05: 65 %) als große (α=0,50: 15 %, davon 5 % katastrophal – eine verrauscht überschätzte Zelle direkt über der Klippe wird zur "gierigen" Wahl, mit Kosten über 100 gegenüber dem Optimum). Eine hohe Lernrate ist dabei nicht schneller: dieselbe Episodenzahl reicht ihr seltener zum Optimum. | `test_alpha_experiment` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Training** | Eine harmlos aussehende Gewohnheit (z. B. am Start gegen die Wand laufen) kann genauso billig geschätzt werden wie die echte Optimalroute – die gierige Policy bleibt dann bei ihr hängen. | Mehr Episoden |
| **GLIE (Epsilon fällt gegen null)** | Ohne Zerfall bleibt das Verhalten dauerhaft zufällig – die Erfahrung konzentriert sich nie auf die produktive Gegend. | – |
| **Eine sinnvolle Lernrate** | Zu groß: die Q-Schätzung überschreibt sich fast vollständig mit jeder neuen (verrauschten) Erfahrung – eine zufällig überschätzte Zelle direkt über der Klippe kann so zur "gierigen" Wahl werden. | – |
| **On-Policy-Verhalten während des Lernens ist irrelevant** | Q-Learnings Zielwert ist immer die gierige Aktion, auch wenn während des Trainings epsilon-gierig (also riskant) gehandelt wird. | SARSA (Stück 4): lernt die Policy, die tatsächlich befolgt wird |
| **Endlich viele States und Actions (Tabelle)** | Ein sehr großes oder stetiges Raster macht eine dichte Q-Tabelle unhandlich. | Funktionsapproximation / DQN (Stück 6) |
| **Jede Erfahrung wird nur einmal genutzt, dann verworfen** | Teuer gesammelte Erfahrung wird nicht wiederverwendet. | Dyna-Q (Stück 5): lernt zusätzlich ein Modell und plant damit |

## Tests

`tests/` prüft das Vehikel (`ql_grid.py`: dasselbe Übergangsmodell wie `value-iteration-demo`, dazu `step` von Hand nachgerechnet inklusive der Rutsch-Schwellen und ein statistischer Abgleich der Sprunghäufigkeiten gegen das Modell), die Referenzlösung (`ql_reference.py`: Bellman-Formel von Hand, entarteter Ein-Zeilen-Fall mit geschlossener Lösung), den Agenten (`ql_agent.py`: TD-Update von Hand für terminale und nicht-terminale Übergänge, GLIE-Epsilon-Schema, gierige/erkundende Aktionswahl mit Gleichstand-Auflösung, eine unabhängige Schritt-für-Schritt-Nachrechnung von `run_episode` UND ein struktureller Kürzungstest – ein Trainings-Snapshot nach $k$ Episoden muss exakt einem separaten Lauf mit nur $k$ Episoden entsprechen), die Auswertung und drei Experimente, die Presets und Permalinks, die Plotly-Achsen (ein im Browser gefundener Bug: deutsch formatierte Dezimal-Strings wie "0,10" wurden von Plotly als tausendergruppierte Ganzzahlen fehlinterpretiert, siehe `test_visualization.py`), die App (AppTest: Standard, jedes Preset, Trainingsstand-Slider, Permalink-Klemmen/-Einrasten, Extremwerte, drei Experimente auf Abruf) und jede Zahl dieses READMEs (`test_claims.py`). Q-Learning ist stochastisch – Einzelläufe sind exakt (fester Seed), Mehr-Seed-Aussagen tragen großzügige Bänder. 65 Tests, Laufzeit gut zwei Minuten (die drei Experimente in `test_claims.py` messen mit vollen Seed-Zahlen); die CI läuft bei jedem Push und wöchentlich.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Episode-für-Episode-Ansicht, Kernfrage, drei Experimente auf Abruf, Grenzen, Formeln |
| `ql_constants.py` | Regler-Grenzen, feste Rewards, Lernparameter, Experimentkonstanten |
| `ql_grid.py` | Das Vehikel: Raster, Zellenarten, `step` (Einzelübergang für den Agenten), `build_model` (nur für die Referenz) |
| `ql_agent.py` | Q-Learning: epsilon-gierige Aktionswahl, TD-Update, Trainingsschleife mit Snapshots |
| `ql_reference.py` | Value Iteration und Policy Evaluation – nur zur Gegenprobe, nie vom Agenten benutzt |
| `ql_evaluation.py` | Analyse, drei Experimente |
| `ql_visualization.py` | Plotly-Abbildungen (Raster als Heatmap mit Policy-Pfeilen, alle Achsen gesperrt) |
| `ql_presets.py` | Presets, Permalink |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **On-Policy-Lernen** (die tatsächlich befolgte, nicht die gierige Policy lernen) – das ist SARSA (Stück 4).
- **Wiederverwendung der gesammelten Erfahrung** (ein Modell lernen und damit planen) – das ist Dyna-Q (Stück 5).
- **Funktionsapproximation** – die Q-Tabelle bleibt hier dicht und klein genug, um sie vollständig zu speichern (Stück 6).

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly und numpy.
