import numpy as np
import matplotlib.pyplot as plt
from histeresis import HisteresisHierroSilicio

def plot_histeresis():
    # Instancia de la clase HisteresisHierroSilicio
    histeresis = HisteresisHierroSilicio()

    # Parámetros para la simulación
    masa = 1e-2  # Masa del núcleo (kg)
    H_max = 25  # Valor máximo del campo magnético H (A/m)
    H_values = np.linspace(-H_max, H_max, 1000)  # Aumentar la resolución con 1000 puntos

    # Calcular B para cada valor de H
    B_values = [histeresis.B(H, masa) for H in H_values]

    # Graficar la curva de histéresis
    plt.figure(figsize=(10, 8))  # Aumentar el tamaño de la figura
    plt.plot(H_values, B_values, label=f"Curva de histéresis (masa={masa:.1e} kg)", color="blue", linewidth=2)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Curva de Histéresis", fontsize=16)
    plt.xlabel("Campo magnético H (A/m)", fontsize=14)
    plt.ylabel("Inducción magnética B (T)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Ajustar la escala del eje x
    plt.xticks(np.arange(-H_max, H_max + 1, H_max / 5), fontsize=12)  # Dividir el eje x en 5 intervalos grandes
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_histeresis()
