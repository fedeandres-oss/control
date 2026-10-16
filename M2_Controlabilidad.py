import numpy as np
import sympy as sp

np.set_printoptions(precision=6, suppress=True)

# ==========================================
# 1. DEFINICIÓN DEL SISTEMA BASE (3x3)
# ==========================================
A = np.array([[-4000.1, -25400, 0],
              [1, 0, 0],
              [0, 1, 0]], dtype=float)

B = np.array([[1],
              [0],
              [0]], dtype=float)

C = np.array([[0, 0, 50000]], dtype=float)

n = A.shape[0]

# ==========================================
# 2. VERIFICACIÓN DE CONTROLABILIDAD (Co)
# ==========================================
Co = np.zeros((n, n))
for j in range(n):
    Co[:, j] = np.linalg.matrix_power(A, j) @ B.ravel()

rCo = np.linalg.matrix_rank(Co)
dCo = np.linalg.det(Co)

print("\n--- 1. VERIFICACIÓN DE CONTROLABILIDAD ---")
print(f"Rango(Co) = {rCo} / {n} | Det(Co) = {dCo:.6f}")

if rCo < n:
    raise ValueError("El sistema NO es controlable.")
else:
    print("¡El sistema es CONTROLABLE!")

# ==========================================
# 3. POLINOMIO EN LAZO ABIERTO (Pla)
# ==========================================
coef_Pla = np.poly(A)
alpha = coef_Pla[1:]

s = sp.Symbol('s')
Pla_simbolico = sum(c * s**(n - i) for i, c in enumerate(coef_Pla))

# ==========================================
# 4. PARÁMETROS DESEADOS Y CÁLCULO DE Pd
# ==========================================
zeta = 0.4559
wn   = 4.3869

P_dom_sym = s**2 + 2*zeta*wn*s + wn**2

if n > 2:
    P_lej_sym = (s + 10*zeta*wn)**(n - 2)
    Pd_simbolico = sp.expand(P_dom_sym * P_lej_sym)
else:
    Pd_simbolico = sp.expand(P_dom_sym)

coef_Pd = np.array([float(c) for c in sp.Poly(Pd_simbolico, s).all_coeffs()])
d = coef_Pd[1:]            # [d1, d2, ..., dn]

# ==========================================
# 5. CONSTRUCCIÓN CORRECTA DE LA MATRIZ T
# ==========================================
T = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        T[i, j] = coef_Pla[j - i]

# ==========================================
# 6. MATRIZ P Y GANANCIA AUXILIAR K_aux
# ==========================================
P = Co @ T
K_aux = np.array([[d[k] - alpha[k] for k in range(n)]])

# ==========================================
# 7. CÁLCULO DE GANANCIA REAL K DEL CONTROLADOR
# ==========================================
K = K_aux @ np.linalg.inv(P)

# ==========================================
# IMPRESIÓN DE RESULTADOS
# ==========================================
print("\n====================================================")
print("  CONTROLADOR POR MÉTODO DE LA MATRIZ T Y P (UMNG)")
print("====================================================")

print("\nPolinomio en lazo abierto Pla(s):")
sp.pprint(sp.N(Pla_simbolico, 6))

print("\nMatriz T:")
print(T)

print("\nMatriz P = Co * T:")
print(P)

print("\nGanancia Auxiliar K_barra:")
print(K_aux)

print("\nGanancia Real del Controlador K:")
print(K)

print("\nVerificación - coef. de det(sI - (A - BK)):")
print(np.poly(A - B @ K))