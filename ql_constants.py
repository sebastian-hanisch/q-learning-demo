"""Konstanten der Demo "Q-Learning" (Stück 3 der Reinforcement-Learning-Linie): dasselbe Raster wie `value-iteration-demo`, dazu die Lernparameter."""

EPS = 1e-9
SEED_MAX = 999999

# --- Das Raster (wie value-iteration-demo, Cliff-Walking-Vorlage) ----------------------------------------------------------------------------------
STEP_COST = -1.0
CLIFF_PENALTY = -100.0
GOAL_REWARD = 10.0

ROWS_MIN, ROWS_MAX, DEFAULT_ROWS = 3, 6, 4
COLS_MIN, COLS_MAX, DEFAULT_COLS = 4, 12, 8
SLIP_MIN, SLIP_MAX, SLIP_STEP, DEFAULT_SLIP = 0.0, 0.30, 0.02, 0.10
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.80, 0.99, 0.01, 0.95

# --- Referenzlösung (Value Iteration, nur zur Gegenprobe) -------------------------------------------------------------------------------------------
VI_TOL = 1e-8
VI_MAX_ITER = 5000

# --- Lernparameter (Q-Learning: Off-Policy-TD-Kontrolle) --------------------------------------------------------------------------------------------
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP, DEFAULT_ALPHA = 0.05, 0.50, 0.05, 0.10
EPSILON_START_MIN, EPSILON_START_MAX, EPSILON_START_STEP, DEFAULT_EPSILON_START = 0.20, 1.00, 0.05, 1.00
EPSILON_DECAY_MIN, EPSILON_DECAY_MAX, EPSILON_DECAY_STEP, DEFAULT_EPSILON_DECAY = 0.0, 0.02, 0.001, 0.005
EPSILON_MIN = 0.01

EPISODES_MIN, EPISODES_MAX, EPISODES_STEP, DEFAULT_EPISODES = 100, 4000, 100, 1500
MAX_STEPS_PER_EPISODE = 400

# --- Experimente (feste Konfigurationen) ------------------------------------------------------------------------------------------------------------
# EXP_SEEDS bewusst kleiner als in bandit-demo/value-iteration-demo: ein einzelner Trainingslauf ist hier viel teurer (bis zu einigen tausend
# Umgebungsschritten statt einer einzelnen Rechnung), 60 Seeds je Stufe waeren fuer eine "Dauer wenige Sekunden" zu langsam (gemessen).
EXP_SEEDS = 20
EXP_EPISODES = 800
EXP_ALPHA_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.50)
EXP_EPSILON_DECAY_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.02)
EXP_EPISODE_CHECKPOINTS = (100, 300, 600, 1000, 1500, 2500, 4000)

# --- Bewertungsschwellen fuer den Wert-Abstand V*(start) - V_policy(start) ------------------------------------------------------------------------
NEAR_OPTIMAL_GAP = 0.5
CLIFF_GAP_THRESHOLD = 50.0
