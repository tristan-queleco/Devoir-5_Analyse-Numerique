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

    def make_F(ti, h, yi, dyi):
        ti_next = ti + h
        def F(x):
            q_next  =   x[0]*ti_next**3 +   x[1]*ti_next**2 + x[2]*ti_next + x[3]
            dq_next = 3*x[0]*ti_next**2 + 2*x[1]*ti_next    + x[2]
            return [
                x[0]*ti**3    + x[1]*ti**2 + x[2]*ti + x[3] - yi,
                3*x[0]*ti**2  + 2*x[1]*ti  + x[2]           - dyi,
                6*x[0]*ti     + 2*x[1]                      - f(ti, yi, dyi),
                6*x[0]*ti_next + 2*x[1]                     - f(ti_next, q_next, dq_next)
            ]
        return F

    # Iteration 0 : yi=alpha, dyi=beta
    F0 = make_F(ti, h, alpha, beta)
    sol = fsolve(F0, [1, 1, beta, alpha])
    coeffs.append(sol)

    for i in range(N):
        C  = coeffs[i]
        ti = ti + h
        yi  =   C[0]*ti**3 +   C[1]*ti**2 + C[2]*ti + C[3]
        dyi = 3*C[0]*ti**2 + 2*C[1]*ti    + C[2]

        Fi  = make_F(ti, h, yi, dyi)
        sol = fsolve(Fi, C)
        coeffs.append(sol)

    return coeffs