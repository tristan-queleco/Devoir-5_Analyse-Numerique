import numpy as np
import matplotlib.pyplot as plt
from rk4 import rk4

#QESTION i)
a = np.pi / 4
b = 3
N = 16
h = (b - a) / N

u1_0 = (np.sqrt(2) / 2) * np.exp(np.pi / 4)
u2_0 = np.sqrt(2) * np.exp(np.pi / 4)
y0 = np.array([u1_0, u2_0])

def f(t, u):
    u1, u2 = u[0], u[1]
    du1 = u2
    du2 = (2 * np.exp(-t) * np.sin(t)) / (u1 * u2) - 2 * np.exp(t) * np.sin(t)
    return np.array([du1, du2])

#Résolution RK4
ti, yi = rk4(f, a, b, y0, h)


y_rk4 = yi[0]


t_exact = np.linspace(a, b, 1000)
y_exact = np.exp(t_exact) * np.sin(t_exact)

# Graphique 3
plt.figure(3)
plt.plot(t_exact, y_exact, 'k-', label="Solution exacte $y(t) = e^t \sin(t)$")
plt.plot(ti, y_rk4, 'b--o', markersize=4, label="RK4 ($N=16$)")
# La spline S(t) sera ajoutée ici après splines_edo_implicite
plt.xlabel("$t$")
plt.ylabel("$y(t)$")
plt.title("Figure 3 — Comparaison RK4, Spline cubique et solution exacte")
plt.legend()
plt.grid(True)
plt.show()