import numpy as np
import sympy as sp

E = 0.4559
wn = 4.3869
np.set_printoptions(suppress=True, precision=4)


def redondear_expr(expr, decimales=4):
    """Redondea todos los números flotantes dentro de una expresión simbólica de sympy."""
    return expr.xreplace({n: round(float(n), decimales) for n in expr.atoms(sp.Float)})


def polinomio_deseado(E, wn, s, n):
    # Dinámico según el tamaño de la matriz n (para n=3, n-2 = 1)
    pd = (s**2 + 2 * E * wn * s + wn**2) * (s + 10 * E * wn) ** (n - 2)
    return sp.expand(pd)


def matriz_observabilidad(A, C):
    n = A.shape[0]
    O = C.copy()
    C_Ak = C.copy()
    for i in range(1, n):
        C_Ak = C_Ak @ A
        O = np.vstack((O, C_Ak))
    return O


def matriz_controlabilidad(A, B):
    """Co = [B, AB, A^2B, ..., A^(n-1)B]"""
    n = A.shape[0]
    Co = B.copy()
    A_kB = B.copy()
    for i in range(1, n):
        A_kB = A @ A_kB
        Co = np.hstack((Co, A_kB))
    return Co


def polinomio_caracteristico(A):
    """Pla = det(sI - A) simbólico (método de Berkowitz, sin fracciones)."""
    n = A.shape[0]
    s = sp.symbols('s')
    A_sym = sp.Matrix(A)
    Pla = (s * sp.eye(n) - A_sym).det(method='berkowitz')
    return s, sp.expand(Pla)


def ganancia_canonica_controlable(Pla, Pd, s):
    """
    Para A, B en forma canónica controlable:
    K = [d1 - alpha1, d2 - alpha2, ..., dn - alpha_n]
    """
    alpha = np.array([float(c) for c in sp.Poly(Pla, s).all_coeffs()])  # [1, alpha1, ..., alpha_n]
    d = np.array([float(c) for c in sp.Poly(Pd, s).all_coeffs()])       # [1, d1, ..., d_n]
    K = (d[1:] - alpha[1:]).reshape(1, -1)
    return K, alpha, d


def imprimir_matriz(nombre, M):
    print(f"{nombre}:")
    for fila in M:
        valores = "  ".join(f"{v:8.4f}" for v in fila)
        print(f"  [ {valores} ]")


# ------------------- EJEMPLO DE USO -------------------
if __name__ == "__main__":
    # Sistema en forma canónica controlable (3x3)
    A = np.array([[-4000.1 , -25400, 0],
                  [1, 0, 0],
                  [0, 1, 0]], dtype=float)

    B = np.array([[1],
                  [0],
                  [0]], dtype=float)

    C = np.array([[0, 0, 50000]], dtype=float)

    n = A.shape[0]

    # ---------- OBSERVABILIDAD ----------
    print("=" * 50)
    print("OBSERVABILIDAD")
    print("=" * 50)

    O = matriz_observabilidad(A, C)
    imprimir_matriz("Matriz de Observabilidad (O)", O)

    det_O = np.linalg.det(O)
    es_observable = np.linalg.matrix_rank(O) == n

    print(f"\nDeterminante de O : {det_O:.4f}")
    print(f"¿Es observable?   : {es_observable}")

    # ---------- CONTROLABILIDAD ----------
    print("\n" + "=" * 50)
    print("CONTROLABILIDAD")
    print("=" * 50)

    Co = matriz_controlabilidad(A, B)
    imprimir_matriz("Matriz de Controlabilidad (Co)", Co)

    det_Co = np.linalg.det(Co)
    es_controlable = np.linalg.matrix_rank(Co) == n

    print(f"\nDeterminante de Co : {det_Co:.4f}")
    print(f"¿Es controlable?   : {es_controlable}")

    # ---------- POLINOMIO LAZO ABIERTO ----------
    print("\n" + "=" * 50)
    print("POLINOMIO CARACTERISTICO (LAZO ABIERTO)")
    print("=" * 50)

    s, Pla = polinomio_caracteristico(A)
    print(f"Pla = {redondear_expr(Pla)}")

    # ---------- POLINOMIO DESEADO ----------
    print("\n" + "=" * 50)
    print("POLINOMIO DESEADO")
    print("=" * 50)

    Pd = polinomio_deseado(E, wn, s, n)   # Se pasa 'n' para ajustar el grado correctamente
    print(f"Pd = {redondear_expr(Pd)}")

    # ---------- GANANCIA K ----------
    print("\n" + "=" * 50)
    print("GANANCIA K (FORMA CANÓNICA CONTROLABLE)")
    print("=" * 50)

    if not es_controlable:
        raise ValueError("El sistema NO es controlable.")

    K, alpha, d = ganancia_canonica_controlable(Pla, Pd, s)
    print(f"alpha (lazo abierto) : {alpha}")
    print(f"d     (deseado)      : {d}")
    print(f"K = [d - alpha]      : {K}")

    # ---------- VERIFICACIÓN ----------
    print("\nVerificación det(sI - (A - BK)):")
    print(np.poly(A - B @ K))
    print("Polos de lazo cerrado:")
    print(np.linalg.eigvals(A - B @ K))