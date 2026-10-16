import numpy as np

# 1. Definición de la planta original (Funciona para cualquier n x n)
A = np.array([[-4000.1, -25400, 0],
              [1, 0, 0],
              [0, 1, 0]], dtype=float)

B = np.array([[1],
              [0],
              [0]], dtype=float)

C = np.array([[0, 0, 50000]], dtype=float)
n = A.shape[0]  # Dimensión del estado original

# 2. Menú de Selección de Entrada
print("========================================")
print("  SELECCIÓN DE SERVO SISTEMA AUMENTADO  ")
print("========================================")
print("1. Entrada Escalón  (1 integrador)")
print("2. Entrada Rampa    (2 integradores)")
print("3. Entrada Parábola (3 integradores)")
print("========================================")

opcion = int(input("Seleccione una opción (1-3): "))


# Función para calcular la matriz de controlabilidad Co = [B AB A^2B ... A^(dim-1)B]
def calcular_ctrb(A_mat, B_mat):
    dim = A_mat.shape[0]
    bloques = [B_mat]
    for i in range(1, dim):
        bloques.append(np.linalg.matrix_power(A_mat, i) @ B_mat)
    return np.hstack(bloques)


# Función para calcular la matriz de observabilidad Ob = [C; C*A; C*A^2; ...; C*A^(dim-1)]
def calcular_obsv(A_mat, C_mat):
    dim = A_mat.shape[0]
    bloques = [C_mat]
    for i in range(1, dim):
        bloques.append(C_mat @ np.linalg.matrix_power(A_mat, i))
    return np.vstack(bloques)


# Función para formatear matrices con 4 decimales en formato de corchetes
def formatear_matriz(matriz):
    filas = []
    for fila in matriz:
        elementos = " ".join(f"{val:10.4f}" for val in fila)
        filas.append(f"  [{elementos}]")
    return "[\n" + "\n".join(filas) + "\n]"


def mostrar_formato_python(titulo, Aa, Ba, Ca, E):
    # Verificación de controlabilidad
    Co = calcular_ctrb(Aa, Ba)
    rango_co = np.linalg.matrix_rank(Co)
    
    # Verificación de observabilidad
    Ob = calcular_obsv(Aa, Ca)
    rango_ob = np.linalg.matrix_rank(Ob)
    
    dim = Aa.shape[0]

    print(f"\n# ========================================")
    print(f"#   {titulo} (Dimensión: {dim}x{dim})")
    print(f"# ========================================\n")

    print(f"Aa = {formatear_matriz(Aa)}\n")
    print(f"Ba = {formatear_matriz(Ba)}\n")
    print(f"Ca = {formatear_matriz(Ca)}\n")
    print(f"E  = {formatear_matriz(E)}\n")

    print(f"# Controlabilidad: Rango {rango_co} de {dim} -> ", end="")
    if rango_co == dim:
        print("SISTEMA CONTROLABLE")
    else:
        print("SISTEMA NO CONTROLABLE")

    print(f"# Observabilidad:  Rango {rango_ob} de {dim} -> ", end="")
    if rango_ob == dim:
        print("SISTEMA OBSERVABLE")
    else:
        print("SISTEMA NO OBSERVABLE")


# 3. Cálculo dinámico según la opción elegida
if opcion == 1:
    # --- ESCALÓN (1 integrador: k = 1) ---
    k = 1
    fila1 = np.hstack([A, np.zeros((n, k))])
    fila2 = np.hstack([-C, np.zeros((k, k))])
    Aa = np.vstack([fila1, fila2])

    Ba = np.vstack([B, np.zeros((k, 1))])
    Ca = np.hstack([C, np.zeros((1, k))])
    
    E = np.zeros((n + k, 1))
    E[-1, 0] = 1.0

    mostrar_formato_python("ENTRADA ESCALÓN", Aa, Ba, Ca, E)

elif opcion == 2:
    # --- RAMPA (2 integradores: k = 2) ---
    k = 2
    fila1 = np.hstack([A, np.zeros((n, k))])
    fila2 = np.hstack([np.zeros((1, n)), [[0, 1]]])
    fila3 = np.hstack([-C, np.zeros((1, k))])
    Aa = np.vstack([fila1, fila2, fila3])

    Ba = np.vstack([B, np.zeros((k, 1))])
    Ca = np.hstack([C, np.zeros((1, k))])
    
    E = np.zeros((n + k, 1))
    E[-1, 0] = 1.0

    mostrar_formato_python("ENTRADA RAMPA", Aa, Ba, Ca, E)

elif opcion == 3:
    # --- PARÁBOLA (3 integradores: k = 3) ---
    k = 3
    fila1 = np.hstack([A, np.zeros((n, k))])
    fila2 = np.hstack([np.zeros((1, n)), [[0, 1, 0]]])
    fila3 = np.hstack([np.zeros((1, n)), [[0, 0, 1]]])
    fila4 = np.hstack([-C, np.zeros((1, k))])
    Aa = np.vstack([fila1, fila2, fila3, fila4])

    Ba = np.vstack([B, np.zeros((k, 1))])
    Ca = np.hstack([C, np.zeros((1, k))])
    
    E = np.zeros((n + k, 1))
    E[-1, 0] = 1.0

    mostrar_formato_python("ENTRADA PARÁBOLA", Aa, Ba, Ca, E)

else:
    print("Opción no válida. Ingrese un número del 1 al 3.")