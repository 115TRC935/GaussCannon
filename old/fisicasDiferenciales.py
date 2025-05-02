import numpy as np
import matplotlib.pyplot as plt

# --- Constantes del sistema ---
mu0 = 4 * np.pi * 1e-7  # permeabilidad del vacío (H/m)
rho_material = 7850     # densidad hierro-silicio (kg/m3)
rho_cobre = 1.68e-8     # resistividad cobre (Ohm.m)

# --- Parámetros ---
V0 = 400                # voltios
C = 470e-6              # faradios
m = 0.01                # masa del núcleo (kg)
R = 0.2                 # resistencia (Ohm)
N = 200                 # vueltas
largo_inductor = 0.04   # longitud bobina (m)
diametro_interno = 0.005  # m
diametro_alambre = 0.0005 # 0.5 mm

# --- Simulación ---
dt = 1e-6               # paso de tiempo (1 microsegundo)
t_max = 0.3           # simulación máxima 30ms

# --- Propiedades derivadas ---
area_nucleo = np.pi * (diametro_interno/2)**2
volumen_nucleo = m / rho_material
resistencia_cobre = rho_cobre * (N * (largo_inductor / N)) / (np.pi * (diametro_alambre/2)**2)
R_total = R + resistencia_cobre

# --- Inicialización de variables ---
n_pasos = int(t_max / dt)
tiempo = np.linspace(0, t_max, n_pasos)
I = np.zeros(n_pasos)
Q = np.zeros(n_pasos)
V_C = np.zeros(n_pasos)
V_L = np.zeros(n_pasos)
V_R = np.zeros(n_pasos)
x = np.zeros(n_pasos)
v = np.zeros(n_pasos)
L = np.zeros(n_pasos)
energia_C = np.zeros(n_pasos)
energia_L = np.zeros(n_pasos)
energia_R = np.zeros(n_pasos)
energia_mecanica = np.zeros(n_pasos)
energia_total = np.zeros(n_pasos)
H_campo = np.zeros(n_pasos)
B_campo = np.zeros(n_pasos)

# --- Funciones auxiliares ---
def mu_r_efectiva(x_nucleo):
    if abs(x_nucleo) <= largo_inductor/2:
        return 5000
    elif abs(x_nucleo) <= largo_inductor*1.5:
        return 1 + (5000-1)*(1 - (abs(x_nucleo)-largo_inductor/2)/(largo_inductor))
    else:
        return 1

def inductancia(x_nucleo):
    return mu0 * mu_r_efectiva(x_nucleo) * (N**2) * area_nucleo / largo_inductor

def dLdx(x_nucleo):
    delta = 1e-5
    return (inductancia(x_nucleo + delta) - inductancia(x_nucleo - delta)) / (2 * delta)

# --- Condiciones iniciales ---
Q[0] = C * V0
V_C[0] = V0
E_inicial = 0.5 * C * V0**2

# --- Simulación paso a paso ---
for i in range(n_pasos-1):
    L[i] = inductancia(x[i])

    V_R[i] = R_total * I[i]
    V_L_extra = I[i] * dLdx(x[i]) * v[i]
    dI_dt = (V_C[i] - V_R[i] - V_L_extra) / L[i]
    I[i+1] = I[i] + dI_dt * dt

    Q[i+1] = Q[i] - I[i] * dt
    V_C[i+1] = Q[i+1] / C

    fuerza = 0.5 * I[i]**2 * abs(dLdx(x[i]))

    a = fuerza / m
    v[i+1] = v[i] + a * dt
    x[i+1] = x[i] + v[i] * dt

    energia_C[i+1] = 0.5 * C * V_C[i+1]**2
    energia_L[i+1] = 0.5 * L[i] * I[i+1]**2
    energia_R[i+1] = energia_R[i] + (R_total * I[i]**2) * dt
    energia_mecanica[i+1] = 0.5 * m * v[i+1]**2
    energia_total[i+1] = energia_C[i+1] + energia_L[i+1] + energia_R[i+1] + energia_mecanica[i+1]

    H_campo[i] = N * I[i] / largo_inductor
    B_campo[i] = mu0 * mu_r_efectiva(x[i]) * H_campo[i]

    # Condición de corte
    if energia_C[i+1] + energia_L[i+1] < 0.0001 * E_inicial:
        n_pasos = i + 2
        break

# Recorte arrays
tiempo = tiempo[:n_pasos]
I = I[:n_pasos]
Q = Q[:n_pasos]
V_C = V_C[:n_pasos]
V_R = V_R[:n_pasos]
V_L = V_L[:n_pasos]
x = x[:n_pasos]
v = v[:n_pasos]
L = L[:n_pasos]
energia_C = energia_C[:n_pasos]
energia_L = energia_L[:n_pasos]
energia_R = energia_R[:n_pasos]
energia_mecanica = energia_mecanica[:n_pasos]
energia_total = energia_total[:n_pasos]
H_campo = H_campo[:n_pasos]
B_campo = B_campo[:n_pasos]

# --- Graficar resultados ---
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(tiempo*1000, energia_total, label='Energía Total (J)')
plt.plot(tiempo*1000, energia_C, label='Energía Capacitor (J)')
plt.plot(tiempo*1000, energia_L, label='Energía Inductor (J)')
plt.plot(tiempo*1000, energia_R, label='Energía Disipada (J)')
plt.plot(tiempo*1000, energia_mecanica, label='Energía Cinética (J)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Energía (J)')
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(tiempo*1000, B_campo, label='Campo B (T)')
plt.plot(tiempo*1000, H_campo, label='Campo H (A/m)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Campo')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# --- Reporte final ---
print("\n=== Reporte Final ===")
print(f"Duración simulada: {tiempo[-1]*1000:.2f} ms")
print(f"Energía inicial: {E_inicial:.5f} J")
print(f"Energía final total: {energia_total[-1]:.5f} J")
print(f"Energía cinética final: {energia_mecanica[-1]:.5f} J")
print(f"Energía disipada en resistencia: {energia_R[-1]:.5f} J")
print(f"Error de conservación: {100 * abs(energia_total[-1] - E_inicial) / E_inicial:.6f} %")
