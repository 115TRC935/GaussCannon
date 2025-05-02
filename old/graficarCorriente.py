# Archivo: graficarCorriente.py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from main import CircuitoRLC

def graficarCorriente():
    """
    Grafica la corriente I(t) en función del tiempo para el circuito RLC.
    Permite ajustar el rango de tiempo con un slider.
    """
    # Crear una instancia de CircuitoRLC
    circuito = CircuitoRLC()

    # Parámetros del circuito
    L = circuito.L_base
    R = circuito.R
    C = circuito.C
    V0 = circuito.V0

    # Configuración inicial del tiempo de simulación
    t_min = 1e-3  # 1 ms
    t_max = 10    # 10 s
    t_default = 0.1  # Valor inicial del slider (100 ms)
    num_puntos = 1000

    # Función para actualizar la gráfica
    def actualizar_grafico(val):
        t_max_slider = slider.val
        tiempos = [i * t_max_slider / num_puntos for i in range(num_puntos)]
        corrientes = [circuito.corriente(t, L) for t in tiempos]

        linea.set_xdata(tiempos)
        linea.set_ydata(corrientes)
        ax.relim()
        ax.autoscale_view()
        plt.draw()

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.2)  # Espacio para el slider

    # Datos iniciales
    tiempos = [i * t_default / num_puntos for i in range(num_puntos)]
    corrientes = [circuito.corriente(t, L) for t in tiempos]

    # Graficar la corriente inicial
    linea, = ax.plot(tiempos, corrientes, label="Corriente I(t)")
    ax.set_title("Corriente en el circuito RLC")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Corriente (A)")
    ax.grid(True)
    ax.legend()

    # Crear el slider para ajustar el rango de tiempo
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Posición del slider
    slider = Slider(ax_slider, 'Tiempo (s)', t_min, t_max, valinit=t_default, valstep=0.1, valfmt="%.2f s")

    # Conectar el slider con la función de actualización
    slider.on_changed(actualizar_grafico)

    # Mostrar la gráfica
    plt.show()

if __name__ == "__main__":
    graficarCorriente()