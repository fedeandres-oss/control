import numpy as np
import sympy as sp

np.set_printoptions(precision=15)  # equivalente aproximado a "format long g"

# ==========================================
# 1. DEFINICIÓN DEL SISTEMA BASE
# ==========================================
A = np.array([[-4000.1, -25400, 0],
              [1, 0, 0],
              [0, 1, 0]], dtype=float)

B = np.array([[1],
              [0],
              [0]], dtype=float)

C = np.array([[0, 0, 50000]], dtype=float)

n = A.shape[0]  # Dimensión genérica n

# ==========================================
# 2. MATRIZ DE CONTROLABILIDAD (Co)
# ==========================================
Co = np.zeros((n, n))
for j in range(n):
    Co[:, j] = (np.linalg.matrix_power(A, j) @ B).ravel()  # B, A*B, ..., A^(n-1)*B

rCo = np.linalg.matrix_rank(Co)
print("\n--- 1. VERIFICACIÓN DE CONTROLABILIDAD ---")
print(f"Rango(Co) = {rCo} / {n}")

if rCo < n:
    raise ValueError("El sistema NO es controlable. No se puede aplicar la fórmula de Ackermann.")
else:
    print("¡El sistema es CONTROLABLE!")

# ==========================================
# 3. POLINOMIO EN LAZO ABIERTO (Pla) Y POLINOMIO DESEADO (Pd)
# ==========================================
s = sp.Symbol('s')

# --- Polinomio en lazo abierto: det(sI - A) ---
coef_Pla = np.poly(A)  # [1, alpha1, ..., alpha_n]
Pla_simbolico = sum(c * s**(n - i) for i, c in enumerate(coef_Pla))

# --- Polinomio deseado ---
zeta = 0.4559   # Coeficiente de amortiguamiento
wn   = 4.3869   # Frecuencia natural [rad/s]

P_dom_sym = s**2 + 2*zeta*wn*s + wn**2

if n > 2:
    P_lej_sym = (s + 10*zeta*wn)**(n - 2)
    Pd_simbolico = sp.expand(P_dom_sym * P_lej_sym)
else:
    Pd_simbolico = sp.expand(P_dom_sym)

coef_Pd = np.array([float(c) for c in sp.Poly(Pd_simbolico, s).all_coeffs()])
# coef_Pd = [1, a1, a2, ..., an]

# ==========================================
# 4. EVALUACIÓN DE phi(A) - MATRIZ POLINÓMICA
# ==========================================
phi_A = np.zeros((n, n))
# Evaluamos sum( coef_Pd(i) * A^(n - i + 1) )  (i = 1..n+1 en MATLAB)
for i in range(1, n + 2):
    potencia = n - i + 1
    phi_A = phi_A + coef_Pd[i - 1] * np.linalg.matrix_power(A, potencia)

# ==========================================
# 5. CÁLCULO DE LA GANANCIA K SEGÚN FÓRMULA DE ACKERMANN UMNG
# ==========================================
vector_e_n = np.zeros((1, n))
vector_e_n[0, -1] = 1  # Vector fila [0 0 ... 1]

# K = [0 0 ... 1] * inv(Co) * phi(A)
K = vector_e_n @ np.linalg.inv(Co) @ phi_A

# ==========================================
# IMPRESIÓN DE RESULTADOS
# ==========================================
print("\n====================================================")
print("  CONTROLADOR POR ACKERMANN (DIAPOSITIVA UMNG)")
print("====================================================")

print("Polinomio en lazo abierto Pla(s):")
sp.pprint(sp.N(Pla_simbolico, 6))
print("Coeficientes de Pla [1, alpha1, ..., alpha_n]:")
print(coef_Pla)

print("\nPolinomio deseado del controlador Pd(s):")
sp.pprint(sp.N(Pd_simbolico, 6))
print("Coeficientes de Pd [1, a1, a2, ..., an]:")
print(coef_Pd)

print("\nMatriz phi(A):")
print(phi_A)
print("Matriz inv(Co):")
print(np.linalg.inv(Co))
print("Ganancia del Controlador K (Vector Fila):")
print(K)
