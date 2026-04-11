import numpy as np

def rk4(f, t0, tf, y0, h):
    # f : membre de droite f(t,y) (fonction de 2 paramètres)
    # t0 : temps initial
    # tf : temps final
    # y0: condition initiale sur y
    # h : longueur du pas de temps
    # ti : les temps discrétisés
    # yi : une matrice (nbr équations) x (nbr pas de temps) contenant la solution discrétisée à chaque équation

    y0 = np.atleast_1d(np.array(y0, dtype=float))
    ti = [t0]
    yi = [y0]

    while ti[-1] < tf:
        t = ti[-1]
        y = yi[-1]
        k1 = h * f(t,y)
        k2 = h * f(t+h/2,y+k1/2)
        k3 = h * f(t+h/2,y+k2/2)
        k4 = h * f(t+h,y+k3)
        ti.append(t+h)
        yi.append(y+(1/6)*(k1+2*k2+2*k3+k4))

    return np.array(ti), np.array(yi).T