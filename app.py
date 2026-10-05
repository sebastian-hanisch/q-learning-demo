"""Q-Learning - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Reinforcement-Learning-Linie der "Konzepte"-Reihe: derselbe Lagerroboter wie in value-iteration-demo, aber das Modell (Rutsch-
Wahrscheinlichkeit, Rewards) ist jetzt UNBEKANNT - Q-Learning (Watkins & Dayan 1992) lernt die optimale Policy allein aus Erfahrung (Ausprobieren),
ohne je die Bellman-Gleichung direkt auf einem Modell auszuwerten.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import ql_constants as C
from ql_evaluation import Settings, alpha_experiment, analyse, decay_experiment, episodes_experiment
from ql_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, sync_query_params
from ql_reference import value_iteration
from ql_visualization import build_alpha, build_decay, build_episodes_gap, build_grid, build_learning_curve

st.set_page_config(page_title="Q-Learning – Sebastian Hanisch", layout="wide")


def de(x, digits=2):
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _analyse(settings, checkpoints):
    return analyse(settings, checkpoints=checkpoints)


@st.cache_data(show_spinner=False)
def _vi_backups(rows, cols, slip, gamma):
    from ql_grid import Grid, build_model
    grid = Grid(rows, cols, slip, gamma)
    P, R = build_model(grid)
    V = np.zeros(grid.n_states)
    sweeps = 0
    for _ in range(C.VI_MAX_ITER):
        Q = R + gamma * np.einsum("sap,p->sa", P, V)
        V_new = Q.max(axis=1)
        sweeps += 1
        if np.max(np.abs(V_new - V)) < C.VI_TOL:
            break
        V = V_new
    return sweeps * grid.n_states, sweeps


@st.cache_data(show_spinner=False)
def _episodes_exp(base):
    return episodes_experiment(base=base)


@st.cache_data(show_spinner=False)
def _decay_exp(base):
    return decay_experiment(base=base)


@st.cache_data(show_spinner=False)
def _alpha_exp(base):
    return alpha_experiment(base=base)


def _frames(episodes, n=12):
    if episodes <= 0:
        return (0,)
    return tuple(sorted({0} | {int(round(x)) for x in np.linspace(1, episodes, min(episodes, n - 1))}))


st.title("🤖 Q-Learning")
st.markdown(
    """
Derselbe Lagerroboter wie bei **Value Iteration und Policy Iteration** (Stück 2) - Raster, Klippe, Packstation, Rutsch-Wahrscheinlichkeit. Diesmal aber kennt der Roboter das Modell **nicht**: er weiß weder, wie
wahrscheinlich ein Rutscher ist, noch was ihn eine Aktion erwartungsgemäß kostet. **Q-Learning** (Watkins & Dayan 1992) lernt die optimale Policy trotzdem - allein aus wiederholtem Ausprobieren (Episoden), indem
es nach jedem einzelnen Schritt seine Schätzung $Q(s,a)$ ein Stück in Richtung der beobachteten Belohnung plus der besten bekannten Folgeschätzung korrigiert. Alle Daten sind erzeugt; die Rechnung ist in numpy geschrieben.
"""
)
st.caption(
    "Drittes Stück der **Reinforcement-Learning-Linie** der \"Konzepte\"-Reihe: der Nachfolger von Value Iteration/Policy Iteration (Stück 2) - dasselbe Raster, aber jetzt ohne bekanntes Modell. **Bezug zur Bandit-Wurzel "
    "(Stück 1):** dort gab es keinen Zustand, nur eine Entscheidung; hier lernt derselbe Explore/Exploit-Gedanke (epsilon-gierig) eine ganze Politik über mehrere Zustände hinweg."
)

with st.expander("So funktioniert Q-Learning", expanded=True):
    st.markdown(
        r"""
1. **Kein Modell.** Der Agent sieht nur einzelne Übergänge: Zustand $s$, Aktion $a$, Belohnung $r$, Folgezustand $s'$ - nie die Wahrscheinlichkeiten oder erwarteten Belohnungen selbst.
2. **Die TD-Update-Regel** (temporal difference): $Q(s,a) \leftarrow Q(s,a) + \alpha\,\big(r + \gamma\,\max_{a'} Q(s',a') - Q(s,a)\big)$. Der Klammerausdruck ist der TD-Fehler - die Differenz zwischen der neuen Beobachtung und der bisherigen Schätzung.
3. **Off-Policy:** das Verhalten während des Lernens ist **epsilon-gierig** (mit Wahrscheinlichkeit $\varepsilon$ zufällig, sonst die aktuell beste Aktion) - das Lernziel $\max_{a'} Q(s',a')$ ist aber immer die **gierige** Aktion, unabhängig davon, was tatsächlich als nächstes getan wird.
4. **GLIE** (greedy in the limit with infinite exploration): $\varepsilon$ fällt über die Episoden gegen null, aber nie ganz auf null - nur so wird am Ende jeder Zustand oft genug besucht, ohne dass der Agent ewig zufällig handelt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Raster**")
    rows = st.slider("Zeilen", *bounds("rows_slider"), key="rows_slider", help="Höhe des Rasters (die unterste Zeile trägt die Klippe).")
    cols = st.slider("Spalten", *bounds("cols_slider"), key="cols_slider", help="Breite des Rasters (Start unten links, Ziel unten rechts).")
    slip = st.slider("Rutsch-Wahrscheinlichkeit", *bounds("slip_slider"), key="slip_slider", step=C.SLIP_STEP, format="%.2f", help="Dem Agenten unbekannt - er muss sie aus Erfahrung erschließen.")
    gamma = st.slider("Diskontfaktor γ", *bounds("gamma_slider"), key="gamma_slider", step=C.GAMMA_STEP, format="%.2f")
    st.markdown("**Das Lernen**")
    alpha = st.slider("Lernrate α", *bounds("alpha_slider"), key="alpha_slider", step=C.ALPHA_STEP, format="%.2f", help="Wie stark jede neue Erfahrung die bisherige Q-Schätzung überschreibt.")
    eps0 = st.slider("Start-Epsilon", *bounds("epsilon_start_slider"), key="epsilon_start_slider", step=C.EPSILON_START_STEP, format="%.2f", help="Erkundungsanteil zu Beginn des Trainings.")
    decay = st.slider("Epsilon-Zerfall", *bounds("epsilon_decay_slider"), key="epsilon_decay_slider", step=C.EPSILON_DECAY_STEP, format="%.3f", help="0 = Epsilon bleibt konstant (kein GLIE); größer = schnelleres Abklingen Richtung reinem Ausnutzen.")
    episodes = st.slider("Trainingsepisoden", *bounds("episodes_slider"), key="episodes_slider", step=C.EPISODES_STEP, help="Wie viele vollständige Durchläufe (Start bis Ziel) der Agent trainiert.")

sync_query_params({
    "rows_slider": int(rows), "cols_slider": int(cols), "slip_slider": round(float(slip), 3), "gamma_slider": round(float(gamma), 3),
    "alpha_slider": round(float(alpha), 3), "epsilon_start_slider": round(float(eps0), 3), "epsilon_decay_slider": round(float(decay), 4), "episodes_slider": int(episodes),
})

settings = Settings(int(rows), int(cols), round(float(slip), 3), round(float(gamma), 3), round(float(alpha), 3), round(float(eps0), 3), round(float(decay), 4), int(episodes), seed=0)
frames = _frames(settings.episodes)
with st.spinner("Q-Learning trainiert ..."):
    a = _analyse(settings, frames)
grid = a.grid
s0 = grid.state_of(grid.start)
vi_backups, vi_sweeps = _vi_backups(settings.rows, settings.cols, settings.slip, settings.gamma)

# --- Wachsendes Beispiel -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Episode für Episode zur gelernten Policy")
if "ql_frame_idx" not in st.session_state or st.session_state.get("ql_frame_owner") != settings:
    st.session_state["ql_frame_idx"] = len(frames) - 1
    st.session_state["ql_frame_owner"] = settings
step_col, play_col = st.columns([5, 2])
with step_col:
    idx = st.slider("Trainingsstand", 0, len(frames) - 1, key="ql_frame_idx", help="0 = noch untrainiert (Q überall 0).")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _render(i):
    ep = frames[i]
    Q_snap = a.snapshots.get(ep, np.zeros((grid.n_states, 4)))
    policy = Q_snap.argmax(axis=1)
    V_est = Q_snap.max(axis=1)
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        head = "Vor dem Training (Q überall 0)" if ep == 0 else f"Nach {ep} von {settings.episodes} Episoden"
        c1.markdown(f"**{head}**")
        c1.plotly_chart(build_grid(grid, V_est, policy if ep > 0 else None), width="stretch", key=f"ql_grid_{ep}")
        c2.markdown("**Ertrag je Episode (bis hierhin)**")
        c2.plotly_chart(build_learning_curve(a.returns[:ep] if ep > 0 else np.array([0.0]), a.V_star[s0]), width="stretch", key=f"ql_curve_{ep}")


def _play_frames():
    return range(len(frames))


if auto_play:
    import time as _time
    for i in _play_frames():
        _render(i)
        _time.sleep(0.35)
else:
    _render(idx)

st.markdown("---")

# --- Kernfrage -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Konvergiert Q-Learning auf dieselbe Policy wie Value Iteration - zu welchem Preis?")
reachable = [s for s in range(grid.n_states) if grid.rc_of(s) not in grid.cliff and grid.rc_of(s) != grid.goal]
policy_match = float(np.mean(a.policy[reachable] == a.pi_star[reachable]))
mcols = st.columns(4)
mcols[0].metric("V*(Start), exakt (Value Iteration)", de(a.V_star[s0], 2))
mcols[1].metric("V(Start) der gelernten Policy", de(a.V_pi[s0], 2), delta=f"-{de(a.gap, 2)}" if a.gap > 0.005 else "optimal", delta_color="inverse")
mcols[2].metric("Umgebungsschritte", f"{a.env_steps:,}".replace(",", "."), help="Jeder einzelne Schritt (Aktion + Beobachtung) des Agenten über alle Trainingsepisoden.")
mcols[3].metric("Value Iteration brauchte", f"{vi_backups:,}".replace(",", ".") + " Bellman-Backups", help=f"{vi_sweeps} Sweeps × {grid.n_states} States - Value Iteration kennt das Modell und muss nichts ausprobieren.")
if a.gap < C.NEAR_OPTIMAL_GAP:
    st.success(f"✅ Die gelernte Policy erreicht (nahezu) den optimalen Wert (Abstand {de(a.gap, 3)}) - aber dafür {a.env_steps:,}".replace(",", ".") + f" Umgebungsschritte gegenüber {vi_backups:,}".replace(",", ".") + " Bellman-Backups von Value Iteration auf demselben Modell: **Nicht-Wissen kostet Erfahrung.**")
elif a.gap > C.CLIFF_GAP_THRESHOLD:
    st.error(f"❌ Die gelernte Policy ist katastrophal schlecht (Abstand {de(a.gap, 1)}) - vermutlich hat eine verrauschte Q-Schätzung eine Zelle direkt über der Klippe zur \"gierigen\" Wahl gemacht, statt ihr auszuweichen. Mit dieser Lernrate/diesem Seed passiert das messbar oft (siehe das Lernraten-Experiment und die Grenzen-Tabelle).")
else:
    st.warning(f"⚠️ Die gelernte Policy ist brauchbar, aber noch spürbar suboptimal (Abstand {de(a.gap, 2)}) - oft, weil eine harmlos aussehende Gewohnheit (z. B. am Start gegen die Wand laufen statt aktiv von der Klippe wegzugehen) genauso billig aussieht wie die echte Optimalroute. Mehr Trainingsepisoden oder ein anderer Seed verändern das Ergebnis (Q-Learning ist stochastisch).")
g1, g2 = st.columns(2)
with g1:
    st.markdown("##### Gelernte Policy (nach dem vollen Training)")
    st.plotly_chart(build_grid(grid, a.Q.max(axis=1), a.policy), width="stretch", key="ql_final_grid")
with g2:
    st.markdown("##### Optimale Policy (Value Iteration, zum Vergleich)")
    st.plotly_chart(build_grid(grid, a.V_star, a.pi_star), width="stretch", key="ql_vi_grid")
st.caption(f"Übereinstimmung der gierigen Aktion mit der optimalen Policy auf den erreichbaren Zellen (Klippenzellen und Ziel ausgenommen, dort gibt es keine Entscheidung): {pct(policy_match)}. Farbe/Zahl links: gelernter Schätzwert max Q(s,·); rechts: exakter V*(s).")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viel Training braucht es - und wie viel mehr als Value Iteration?")
st.caption(f"Standardraster, {C.EXP_SEEDS} Seeds je Trainingsdauer, α={de(C.DEFAULT_ALPHA,2)}, Epsilon-Zerfall {de(C.DEFAULT_EPSILON_DECAY,3)}. Gezeigt: Wert-Abstand zu V*(Start) und Umgebungsschritte (log) gegen Value Iterations Bellman-Backups. Dauer bis zu einer Minute.")
if st.button("Trainingsdauern durchrechnen", key="episodes_start"):
    st.session_state["episodes_on"] = True
if st.session_state.get("episodes_on"):
    ee = _episodes_exp(Settings())
    vi_b, _ = _vi_backups(C.DEFAULT_ROWS, C.DEFAULT_COLS, C.DEFAULT_SLIP, C.DEFAULT_GAMMA)
    st.plotly_chart(build_episodes_gap(ee, vi_b), width="stretch", key="episodes_chart")
    r_first, r_last = ee["rows"][0], ee["rows"][-1]
    st.warning(
        f"**Befund:** Der Anteil nahezu optimaler Läufe steigt über die gesamte gemessene Spanne nicht mit dem Training, er schwankt zwischen {pct(min(r['frac_near_optimal'] for r in ee['rows']))} und {pct(max(r['frac_near_optimal'] for r in ee['rows']))} ({pct(r_first['frac_near_optimal'])} bei {r_first['episodes']} Episoden, {pct(r_last['frac_near_optimal'])} bei {r_last['episodes']} Episoden) - **mehr Training allein löst das nicht zuverlässig auf**, weil Epsilon inzwischen kaum noch erkundet und eine harmlos aussehende Gewohnheit (siehe Kernfrage) genauso billig geschätzt wird wie die echte Optimalroute. "
        f"Selbst der teuerste Lauf braucht **{de(r_last['env_steps_mean']/vi_b,0)}× so viele** Umgebungsschritte wie Value Iteration Bellman-Backups auf demselben Modell - Q-Learning zahlt für sein Nicht-Wissen mit Erfahrung, nicht mit Rechenzeit."
    )

st.markdown("---")

st.subheader("🔬 Wie wichtig ist GLIE (Epsilon-Zerfall)?")
st.caption(f"Standardraster, {C.EXP_EPISODES} Episoden, {C.EXP_SEEDS} Seeds je Zerfallsrate (0 = Epsilon bleibt bei {de(C.DEFAULT_EPSILON_START,2)}, also rein zufälliges Verhalten während des GESAMTEN Trainings). Dauer bis zu einer Minute.")
if st.button("Epsilon-Zerfallsraten durchrechnen", key="decay_start"):
    st.session_state["decay_on"] = True
if st.session_state.get("decay_on"):
    de_exp = _decay_exp(Settings(episodes=C.EXP_EPISODES))
    st.plotly_chart(build_decay(de_exp), width="stretch", key="decay_chart")
    r0 = de_exp["rows"][0]
    best = min(de_exp["rows"], key=lambda r: r["mean"])
    st.warning(
        f"**Befund:** Ohne Zerfall (Epsilon bleibt konstant bei {de(C.DEFAULT_EPSILON_START,2)}) erreichen nur {pct(r0['frac_near_optimal'])} der Läufe eine nahezu optimale Policy - **obwohl** dieser Fall im Mittel {de(r0['env_steps_mean']/best['env_steps_mean'],1)}× so viele Umgebungsschritte verbraucht wie die beste gemessene Zerfallsrate ({de(best['decay'],3)}: {pct(best['frac_near_optimal'])} nahezu optimal). "
        "Mehr Erfahrung allein ersetzt GLIE nicht: reines Zufallsverhalten konzentriert die Erfahrung nie auf die produktive Gegend um den optimalen Weg."
    )

st.markdown("---")

st.subheader("🔬 Wie stark wirkt die Lernrate α?")
st.caption(f"Standardraster, {C.EXP_EPISODES} Episoden, {C.EXP_SEEDS} Seeds je Lernrate. Gezeigt: Wert-Abstand und Anteil nahezu optimaler Läufe. Dauer bis zu einer Minute.")
if st.button("Lernraten durchrechnen", key="alpha_start"):
    st.session_state["alpha_on"] = True
if st.session_state.get("alpha_on"):
    al_exp = _alpha_exp(Settings(episodes=C.EXP_EPISODES))
    st.plotly_chart(build_alpha(al_exp), width="stretch", key="alpha_chart")
    lo, hi = al_exp["rows"][0], al_exp["rows"][-1]
    st.warning(
        f"**Befund:** Kleine Lernraten (α={de(lo['alpha'],2)}) liefern verlässlicher nahezu optimale Policies ({pct(lo['frac_near_optimal'])}) als große (α={de(hi['alpha'],2)}: {pct(hi['frac_near_optimal'])}) - eine hohe Lernrate überschreibt die Q-Schätzung fast vollständig mit "
        "jeder einzelnen (verrauschten) Erfahrung, statt sie zu mitteln. Schneller ist eine hohe Lernrate hier nicht: dieselbe Episodenzahl reicht ihr seltener zum Optimum."
    )

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Training** | Eine harmlos aussehende Gewohnheit (z. B. am Start gegen die Wand laufen) kann genauso billig geschätzt werden wie die echte Optimalroute - die gierige Policy bleibt dann bei ihr hängen, statt aktiv von der Klippe wegzugehen (gemessen, siehe Kernfrage/Experiment 1). | Mehr Episoden |
| **GLIE (Epsilon fällt gegen null)** | Ohne Zerfall bleibt das Verhalten dauerhaft zufällig - die Erfahrung konzentriert sich nie auf die produktive Gegend (gemessen, Experiment 2). | - |
| **Eine sinnvolle Lernrate** | Zu groß: die Q-Schätzung überschreibt sich fast vollständig mit jeder neuen (verrauschten) Erfahrung - eine zufällig überschätzte Zelle direkt über der Klippe kann so zur "gierigen" Wahl werden (gemessen, Experiment 3: seltene, aber sehr teure Ausreißer). | - |
| **On-Policy-Verhalten während des Lernens ist irrelevant** | Q-Learnings Zielwert ist immer die gierige Aktion, auch wenn während des Trainings epsilon-gierig (also riskant) gehandelt wird - das Verhalten selbst während des Trainings kann näher an der Klippe verlaufen als die gelernte Policy. | SARSA (Stück 4): lernt die Policy, die tatsächlich befolgt wird |
| **Endlich viele States und Actions (Tabelle)** | Ein sehr großes oder stetiges Raster macht eine dichte Q-Tabelle unhandlich. | Funktionsapproximation / DQN (Stück 6) |
| **Jede Erfahrung wird nur einmal genutzt, dann verworfen** | Teuer gesammelte Erfahrung wird nicht wiederverwendet. | Dyna-Q (Stück 5): lernt zusätzlich ein Modell und plant damit |
"""
)
st.caption("Die Linie: Bandit → Value Iteration und Policy Iteration → **Q-Learning** (dieses Stück) → SARSA / Dyna-Q / Funktionsapproximation (DQN) / Policy Gradient → Actor-Critic.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Das Modell** ist identisch zu `value-iteration-demo` (Raster, Rutsch-Wahrscheinlichkeit, Rewards) - der Agent kennt es hier aber nicht; er sieht nur einzelne Übergänge $(s, a, r, s')$ aus `ql_grid.step`.

**TD-Update** (Watkins & Dayan 1992): $Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big)$, initialisiert mit $Q_0 \equiv 0$. Bei einer terminalen Transition (Erreichen des Ziels) entfällt der Bootstrap-Term: $Q(s,a) \leftarrow Q(s,a) + \alpha(r - Q(s,a))$.

**Verhaltenspolicy:** epsilon-gierig, $\varepsilon_e = \max(\varepsilon_{\min}, \varepsilon_0 / (1 + \text{decay} \cdot e))$ in Episode $e$ (GLIE für decay $> 0$).

**Konvergenz** (Watkins & Dayan 1992, unter Standardbedingungen an die Lernrate und GLIE): $Q \to Q^*$ mit Wahrscheinlichkeit 1 bei unendlichem Training - hier nur endlich trainiert, daher die gemessene Lücke zu $V^*$.

Implementiert in `ql_grid.py` (das Vehikel, `step` statt `build_model` für den Agenten), `ql_agent.py` (Q-Learning), `ql_reference.py` (Value Iteration, nur zur Gegenprobe), `ql_evaluation.py` (Analyse, drei Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Reinforcement Learning: Bandit bis Actor-Critic](https://sebastianhanisch.net/konzepte-reinforcement-learning.html)."
)
