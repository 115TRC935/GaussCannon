import time
start_time = time.time()

import numpy as np
import os
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

import cupy as cp

# Constantes físicas
mu0 = 4 * np.pi * 1e-7  # H/m
mu_r = 1.0              # aire
C = 100e-6              # 100 uF
V0 = 10.0               # voltaje inicial (10 V)
rho_cobre = 1.68e-8     # Resistividad del cobre (ohm·m)

# Parámetros discretizados
vueltas = cp.arange(1, 301, 1, dtype=cp.float32)        # 300
d_hilo = cp.arange(0.0001, 0.00201, 0.00005, dtype=cp.float32)  # 39
d_nucleo = cp.arange(0.001, 0.0085, 0.0005, dtype=cp.float32)   # 15
l_nucleo = cp.arange(0.001, 0.0205, 0.0005, dtype=cp.float32)   # 40
l_bobina = cp.arange(0.001, 0.0505, 0.0005, dtype=cp.float32)   # 100

# Crear mallas de parámetros fijos (todo menos vueltas)
D_hilo, D_nucleo, L_nucleo, L_bobina = cp.meshgrid(
    d_hilo, d_nucleo, l_nucleo, l_bobina, indexing='ij'
)
shape_fixed = D_hilo.shape
num_combinations = vueltas.size * np.prod(shape_fixed)
print(f"Total combinaciones: {num_combinations:,}")

# Almacenamiento en CPU
resultados = []

# Precalcular valores que no cambian dentro del ciclo
pi = cp.pi
mu0_mu_r = mu0 * mu_r

block_size = 50
for i in range(0, vueltas.size, block_size):
    v_block = vueltas[i:i+block_size]
    V = v_block[:, None, None, None, None]  # broadcasting

    # Calcular vueltas por capa y capas
    vueltas_por_capa = cp.floor(cp.pi * D_nucleo / D_hilo)
    vueltas_por_capa = cp.maximum(vueltas_por_capa, 1.0)
    capas = cp.ceil(V / vueltas_por_capa)

    # Diámetro medio y área
    d_medio = D_nucleo + D_hilo * capas
    A = pi * (d_medio / 2)**2

    # Inductancia
    L = mu0_mu_r * (V**2) * A / L_bobina  # Henrios

    # Corriente máxima estimada por descarga RLC subamortiguada
    omega_0 = 1 / cp.sqrt(L * C)
    zeta = (rho_cobre * L_bobina) / (2 * cp.sqrt(L / C))  # Esta es una estimación general para zeta
    I_max = (V0 / (L**0.5)) * cp.exp(-zeta * cp.pi)  # estimación simplificada

    # Campo magnético en centro del solenoide (aproximado)
    B = mu0_mu_r * V * I_max / L_bobina

    # Calcular resistencia en función de los parámetros del hilo
    # R = resistividad * longitud / área
    R_calculada = (rho_cobre * L_bobina) / (pi * (D_hilo / 2)**2)  # Ohmios

    # Convertir a NumPy y almacenar planos aplanados
    B_cpu = cp.asnumpy(B).ravel()
    V_cpu = cp.asnumpy(V).ravel()
    L_cpu = cp.asnumpy(L).ravel()
    R_cpu = cp.asnumpy(R_calculada).ravel()
    d_hilo_cpu = cp.asnumpy(D_hilo).ravel()
    d_nucleo_cpu = cp.asnumpy(D_nucleo).ravel()
    l_nucleo_cpu = cp.asnumpy(L_nucleo).ravel()
    l_bobina_cpu = cp.asnumpy(L_bobina).ravel()

    # Guardar todos como lista de tuplas
    for b_val, n_val, l_val, r_val, dh_val, dn_val, ln_val, lb_val in zip(
        B_cpu, V_cpu, L_cpu, R_cpu, d_hilo_cpu, d_nucleo_cpu, l_nucleo_cpu, l_bobina_cpu):
        resultados.append((b_val, n_val, l_val, r_val, dh_val, dn_val, ln_val, lb_val))

    # Liberar memoria
    cp._default_memory_pool.free_all_blocks()

# Ordenar por campo magnético descendente
resultados.sort(reverse=True, key=lambda x: x[0])

# Mostrar los 10 mejores
print("\nTop 10 configuraciones con mayor campo magnético (B):")
print("   B [T]      |  N vueltas  |  L [H]    | R [Ohm]   | d_hilo [m] | d_nucleo [m] | l_nucleo [m] | l_bobina [m]")
print("------------------------------------------------------------------------------------------------------------")
for b, n, l, r, dh, dn, ln, lb in resultados[:10]:
    print(f"{b:.6e}  |   {int(n):4d}       |  {l:.3e}  | {r:.6e} | {dh:.5e}  | {dn:.5e}   | {ln:.5e}   | {lb:.5e}")

# Mostrar el tiempo total de ejecución
print(f"\nTiempo total de ejecución: {time.time() - start_time:.2f} segundos")
