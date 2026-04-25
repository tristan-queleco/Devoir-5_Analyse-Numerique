import numpy as np
import matplotlib.pyplot as plt
from splines_edo_implicite import splines_edo_implicite
from rk4 import rk4

# ── Paramètres du PVI ──────────────────────────────────────────
t0 = np.pi / 4
tf = 3
N  = 16

alpha = (np.sqrt(2) / 2) * np.exp(np.pi / 4)   # y(t0)
beta  = np.sqrt(2) * np.exp(np.pi / 4)          # y'(t0)

# Membre de droite : y'' = f(t, y, dy)
def f(t, y, dy):
    return (2 * np.exp(-t) / np.sin(t)) * y * dy - 2 * np.exp(t) * np.sin(t)

# Solution exacte
def y_exact(t):
    return np.exp(t) * np.sin(t)

# ── Question (c) : spline cubique N=16 ────────────────────────
coeffs = splines_edo_implicite(alpha, beta, f, t0, tf, N)

h = (tf - t0) / N
t_vals = np.linspace(t0, tf, 1000)

# Évaluer la spline S(t)
S_vals = []
for t in t_vals:
    i = min(int((t - t0) / h), N - 1)
    C = coeffs[i]
    S_vals.append(C[0]*t**3 + C[1]*t**2 + C[2]*t + C[3])

# Figure 1
plt.figure()
plt.plot(t_vals, y_exact(t_vals), label="Solution exacte $y(t) = e^t \sin t$", linewidth=2)
plt.plot(t_vals, S_vals, '--', label="Spline cubique $S(t)$", linewidth=2)
plt.xlabel("t")
plt.ylabel("y")
plt.title("Figure 1 : Spline cubique vs solution exacte (N=16)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── Question (d) : erreur globale en échelle log ───────────────
N_vals = [2**6, 2**7, 2**8, 2**9, 2**10]
erreurs = []

for N in N_vals:
    coeffs = splines_edo_implicite(alpha, beta, f, t0, tf, N)
    h = (tf - t0) / N
    t_noeuds = [t0 + i * h for i in range(N + 1)]
    
    E = max(abs(y_exact(t_noeuds[i]) - (
        coeffs[i][0]*t_noeuds[i]**3 + coeffs[i][1]*t_noeuds[i]**2 +
        coeffs[i][2]*t_noeuds[i] + coeffs[i][3]
    )) for i in range(N + 1))
    
    erreurs.append(E)

h_vals = [(tf - t0) / N for N in N_vals]

plt.figure(2)
plt.loglog(h_vals, erreurs, 'o-', label="Erreur globale E(h)")
plt.xlabel("h")
plt.ylabel("E(h)")
plt.title("Figure 2 : Erreur globale en fonction de h (échelle log-log)")
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()