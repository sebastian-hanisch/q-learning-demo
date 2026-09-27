"""Referenzlösung (nur zur Gegenprobe, NIE vom lernenden Agenten benutzt): Value Iteration auf dem bekannten Modell (P, R), wie in `value-iteration-demo`.
Bellman-Optimalitätsgleichung: V*(s) = max_a sum_s' P(s'|s,a) (R(s,a,s') + gamma V*(s'))."""

import numpy as np

import ql_constants as C
from ql_grid import ACTIONS


def q_values(P, R, V, gamma):
    return R + gamma * np.einsum("sap,p->sa", P, V)


def value_iteration(P, R, gamma, tol=C.VI_TOL, max_iter=C.VI_MAX_ITER):
    S, A = R.shape
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = q_values(P, R, V, gamma)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = q_values(P, R, V, gamma)
    policy = Q.argmax(axis=1)
    return V, Q, policy


def policy_evaluation(P, R, policy, gamma, tol=C.VI_TOL, max_iter=C.VI_MAX_ITER):
    """Wahrer Wert einer (z. B. gelernten) Policy unter dem bekannten Modell - die faire Gegenprobe: nicht ob jede Zelle dieselbe Action traegt wie
    die optimale Policy (irrelevant fuer Zellen, die die Policy nie besucht), sondern ob sie ebenso gut ABSCHNEIDET."""
    S = P.shape[0]
    idx = np.arange(S)
    P_pi = P[idx, policy]
    R_pi = R[idx, policy]
    V = np.zeros(S)
    for _ in range(max_iter):
        V_new = R_pi + gamma * (P_pi @ V)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V
