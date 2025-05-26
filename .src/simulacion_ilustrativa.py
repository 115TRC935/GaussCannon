import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
# Parámetros genéricos para ilustración
T = 0.1  # tiempo total (s)
dt = 0.0005
N = int(T/dt)
t = np.linspace(0, T, N)

# Movimiento de la bala (posición vs tiempo) para cada caso
x_sub = 0.5 * (1 - np.exp(-15*t) * np.cos(60*t))
x_crit = 1 - (1 + 30*t) * np.exp(-30*t)
x_sobre = 1 - (1 + 10*t) * np.exp(-10*t)

# Corriente y tensión (ilustrativos)
I_sub = np.exp(-15*t) * np.sin(60*t)
V_sub = np.exp(-15*t) * np.cos(60*t)
I_crit = 30*t * np.exp(-30*t)
V_crit = np.exp(-30*t)
I_sobre = 10*t * np.exp(-10*t)
V_sobre = np.exp(-10*t)

# --- FIGURA CON 3 COLUMNAS: cada columna un caso ---
fig, axs = plt.subplots(3, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [2, 1]})
casos = [
    ('Subamortiguado', x_sub, I_sub, V_sub, 'red'),
    ('Críticamente amortiguado', x_crit, I_crit, V_crit, 'green'),
    ('Sobreamortiguado', x_sobre, I_sobre, V_sobre, 'purple'),
]

# Ejes de animación y gráficos
bala_rects = []
for i, (nombre, x_tray, I, V, color) in enumerate(casos):
    # Movimiento (columna 0)
    ax_mov = axs[i, 0]
    ax_mov.set_xlim(0, 1.2)
    ax_mov.set_ylim(-0.2, 0.6)
    ax_mov.set_yticks([])
    ax_mov.set_xticks([])
    ax_mov.set_title(nombre)
    # Bobinas
    coil1 = plt.Rectangle((0.1, 0), 0.2, 0.5, color='gray', alpha=0.5)
    coil2 = plt.Rectangle((0.5, 0), 0.1, 0.5, color='blue', alpha=0.5)
    ax_mov.add_patch(coil1)
    ax_mov.add_patch(coil2)
    # Bala
    bala = plt.Rectangle((0.05, 0.2), 0.05, 0.1, color=color)
    ax_mov.add_patch(bala)
    bala_rects.append((bala, x_tray))
    # Corriente y tensión (columna 1)
    ax_graf = axs[i, 1]
    ax_graf.plot(t, I, label='Corriente', color='tab:blue')
    ax_graf.plot(t, V, label='Tensión', color='tab:orange')
    ax_graf.set_ylabel('u.a.')
    ax_graf.set_xlabel('Tiempo (s)')
    ax_graf.legend()
    ax_graf.grid()
    if i == 0:
        ax_graf.set_title('Corriente y Tensión')

plt.tight_layout()

# --- ANIMACIÓN DE LAS 3 BALAS EN PARALELO ---
def animate(frame):
    for bala, x_tray in bala_rects:
        bala.set_x(0.05 + x_tray[frame]*0.9)
    return [bala for bala, _ in bala_rects]

ani = animation.FuncAnimation(fig, animate, frames=N, interval=30, blit=True)
ani.save('animacion_amortiguacion.gif', writer=PillowWriter(fps=30))
plt.show()
