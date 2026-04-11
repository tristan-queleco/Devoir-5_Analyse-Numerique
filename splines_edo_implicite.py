from scipy.optimize import fsolve
import numpy as np

def splines_edo_implicite(alpha, beta, f, t0, tf, N):
    # alpha : condition initiale sur y
    # beta : condition initiale sur y'
    # f : membre de droite, f(t, y, dy) (fonction de trois paramètres)
    # t0 : temps initial
    # tf : temps final
    # N : nombre de sous-intervalles
    # coeff : une liste de vecteurs de coefficients, coeff[i] = [ai bi ci di]

    coeffs = []
    h = (tf - t0) / N

    # Iteration 0
    ti = t0

    p_i = lambda C: C[0]*(ti+h)**3 + C[1]*(ti+h)**2 + C[2]*(ti+h) + C[3]
    dp_i = lambda C: 3*C[0]*(ti+h)**2 + 2*C[1]*(ti+h) + C[2]

    F = lambda x: [
        ti**3*x[0]    + ti**2*x[1] + ti*x[2] + x[3] - alpha,
        3*ti**2*x[0]  + 2*ti*x[1]  + x[2]           - beta,
        6*ti*x[0]     + 2*x[1]                      - f(ti, alpha, beta),
        6*(ti+h)*x[0] + 2*x[1]                      - f(ti+h, p_i(x), dp_i(x))
    ]

    # fsolve est une variante de la méthode de Newton
    sol= fsolve(F, [1, 1, beta, alpha])
    coeffs.append(sol)

    for i in range(N):
        C = coeffs[i]
        ti = ti + h
        yi  =   C[0]*ti**3 +   C[1]*ti**2 + C[2]*ti + C[3]
        dyi = 3*C[0]*ti**2 + 2*C[1]*ti    + C[2]

        p_i = lambda C: C[0]*(ti+h)**3 + C[1]*(ti+h)**2 + C[2]*(ti+h) + C[3]
        dp_i = lambda C: 3*C[0]*(ti+h)**2 + 2*C[1]*(ti+h) + C[2]

        # A COMPLETER
        # F = lambda x: ...

        sol = fsolve(F, C)
        coeffs.append(sol)

    return coeffs