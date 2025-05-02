import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from main import CircuitoRLC, μ0, rho_cobre, m, rho_material
from histeresis import HisteresisHierroSilicio  # Import directly

def calcular_energia_circuito_rlc():
    """
    Calcula y visualiza la energía suministrada por el circuito RLC
    utilizando las ecuaciones correctas para la tensión del inductor
    y calculando la energía como la integral de la potencia.
    """
    # Crear instancia del circuito RLC
    circuito = CircuitoRLC(verbose=False)
    
    # Obtener parámetros del circuito
    L = circuito.L_base
    R = circuito.R
    C = circuito.C
    V0 = circuito.V0
    
    # Calcular la constante de tiempo (tau) del circuito
    tau_RL = L / R  # Constante de tiempo del circuito RL
    omega0 = 1 / np.sqrt(L * C)  # Frecuencia natural
    alpha = R / (2 * L)  # Factor de amortiguamiento
    
    # Determinar el tipo de respuesta del circuito
    if alpha < omega0:  # Subamortiguado
        omega_d = np.sqrt(omega0**2 - alpha**2)  # Frecuencia amortiguada
        tau_efectivo = 1 / alpha  # Constante de tiempo efectiva
        tipo_circuito = "Subamortiguado"
    elif alpha == omega0:  # Críticamente amortiguado
        tau_efectivo = 1 / alpha
        tipo_circuito = "Críticamente amortiguado"
    else:  # Sobreamortiguado
        tau_efectivo = 1 / alpha
        tipo_circuito = "Sobreamortiguado"
    
    # Configuración del tiempo para mostrar 5*tau_efectivo
    t_max = 5 * tau_efectivo
    num_puntos = 5000  # Alta resolución para cálculos precisos
    
    # Generar array de tiempo
    tiempos = np.linspace(0, t_max, num_puntos)
    dt = tiempos[1] - tiempos[0]
    
    # Calcular corriente para cada instante de tiempo
    corrientes = np.array([circuito.corriente(t, L) for t in tiempos])
    
    # Calcular derivada de la corriente para la tensión del inductor
    # Usamos diferencias finitas centradas para mayor precisión
    dI_dt = np.zeros_like(corrientes)
    dI_dt[1:-1] = (corrientes[2:] - corrientes[:-2]) / (2 * dt)  # Derivada central
    dI_dt[0] = (corrientes[1] - corrientes[0]) / dt  # Derivada hacia adelante para primer punto
    dI_dt[-1] = (corrientes[-1] - corrientes[-2]) / dt  # Derivada hacia atrás para último punto
    
    # Calcular carga del capacitor y tensiones
    carga_capacitor = np.zeros_like(corrientes)
    carga_capacitor[0] = C * V0  # Carga inicial
    
    # Integrar corriente para obtener carga del capacitor
    for i in range(1, num_puntos):
        carga_capacitor[i] = carga_capacitor[i-1] - corrientes[i-1] * dt
    
    # Calcular componentes de tensión
    V_R = R * corrientes  # Tensión en resistencia: V_R = R*I
    V_L = L * dI_dt       # Tensión en inductor: V_L = L*dI/dt
    V_C = carga_capacitor / C  # Tensión en capacitor: V_C = Q/C
    
    # Tensión total (por ley de Kirchhoff, debe sumar 0 en un circuito cerrado)
    # En t=0, V_C=V0, V_L tiene un impulso inicial y V_R comienza en 0.
    V_total = V_R + V_L + V_C
    
    # Calcular potencia instantánea para cada componente
    # Potencia = Tensión * Corriente
    P_R = V_R * corrientes  # Potencia disipada en resistencia (siempre positiva)
    P_L = V_L * corrientes  # Potencia en inductor (puede ser + o -)
    P_C = V_C * corrientes  # Potencia en capacitor (puede ser + o -)
    
    # Potencia total
    P_total = P_R + P_L + P_C
    
    # Calcular energía mediante integración numérica (regla del trapecio)
    # Energía = ∫ P(t) dt
    E_R = np.trapz(P_R, tiempos)  # Energía disipada en resistencia
    E_L = np.trapz(P_L, tiempos)  # Energía almacenada/liberada por inductor
    E_C = np.trapz(P_C, tiempos)  # Energía almacenada/liberada por capacitor
    E_total = np.trapz(P_total, tiempos)  # Energía total
    
    # Energía inicial almacenada en el capacitor
    E_inicial = 0.5 * C * V0**2
    
    # Crear gráficos
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    # 1. Gráfico de corriente
    axs[0].plot(tiempos, corrientes, 'b-', linewidth=2, label="I(t)")
    axs[0].axvline(x=tau_efectivo, color='r', linestyle='--', label=f"τ = {tau_efectivo:.3f}s")
    for i in range(2, 6):
        axs[0].axvline(x=i*tau_efectivo, color='r', linestyle=':', alpha=0.5)
    axs[0].set_title(f'Corriente en el circuito {tipo_circuito}')
    axs[0].set_ylabel('Corriente (A)')
    axs[0].grid(True)
    axs[0].legend()
    
    # 2. Gráfico de tensiones
    axs[1].plot(tiempos, V_R, 'r-', linewidth=2, label="V_R")
    axs[1].plot(tiempos, V_L, 'g-', linewidth=2, label="V_L")
    axs[1].plot(tiempos, V_C, 'b-', linewidth=2, label="V_C")
    axs[1].axvline(x=tau_efectivo, color='k', linestyle='--')
    axs[1].set_title('Tensiones en componentes del circuito')
    axs[1].set_ylabel('Tensión (V)')
    axs[1].grid(True)
    axs[1].legend()
    
    # 3. Gráfico de potencias
    axs[2].plot(tiempos, P_R, 'r-', linewidth=2, label="P_R")
    axs[2].plot(tiempos, P_L, 'g-', linewidth=2, label="P_L")
    axs[2].plot(tiempos, -P_C, 'b-', linewidth=2, label="P_C (invertida)") # cuando P_C ya viene como negativa
    axs[2].plot(tiempos, P_total, 'k-', linewidth=2, label="P_total")
    axs[2].axvline(x=tau_efectivo, color='k', linestyle='--')
    axs[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[2].set_title('Potencia instantánea')
    axs[2].set_ylabel('Potencia (W)')
    axs[2].grid(True)
    axs[2].legend()
    
    # 4. Gráfico de energía acumulada (integral de la potencia)
    energia_acumulada_R = np.zeros_like(tiempos)
    energia_acumulada_L = np.zeros_like(tiempos)
    energia_acumulada_C = np.zeros_like(tiempos)
    energia_acumulada_total = np.zeros_like(tiempos)
    
    # Calcular energía acumulada en cada punto de tiempo
    for i in range(1, len(tiempos)):
        energia_acumulada_R[i] = np.trapz(P_R[:i+1], tiempos[:i+1])
        energia_acumulada_L[i] = np.trapz(P_L[:i+1], tiempos[:i+1])
        energia_acumulada_C[i] = np.trapz(P_C[:i+1], tiempos[:i+1])
        energia_acumulada_total[i] = np.trapz(P_total[:i+1], tiempos[:i+1])
    
    axs[3].plot(tiempos, energia_acumulada_R, 'r-', linewidth=2, label="E_R")
    axs[3].plot(tiempos, energia_acumulada_L, 'g-', linewidth=2, label="E_L")
    axs[3].plot(tiempos, energia_acumulada_C, 'b-', linewidth=2, label="E_C")
    axs[3].plot(tiempos, energia_acumulada_total, 'k-', linewidth=2, label="E_total")
    axs[3].axvline(x=tau_efectivo, color='k', linestyle='--')
    axs[3].axhline(y=E_inicial, color='m', linestyle='--', label=f"E_inicial = {E_inicial:.2f}J")
    axs[3].set_title('Energía acumulada')
    axs[3].set_xlabel('Tiempo (s)')
    axs[3].set_ylabel('Energía (J)')
    axs[3].grid(True)
    axs[3].legend()
    
    # Ajustar layout
    plt.tight_layout()
    
    # Imprimir resultados
    print(f"ANÁLISIS DEL CIRCUITO RLC:")
    print(f"=========================")
    print(f"Tipo de circuito: {tipo_circuito}")
    print(f"Parámetros:")
    print(f"  R = {R:.4f} Ω")
    print(f"  L = {L:.6e} H")
    print(f"  C = {C:.6e} F")
    print(f"  V0 = {V0:.1f} V")
    print(f"Características dinámicas:")
    print(f"  τ (constante de tiempo) = {tau_efectivo:.6f} s")
    print(f"  α (factor de amortiguamiento) = {alpha:.2f}")
    print(f"  ω₀ (frecuencia natural) = {omega0:.2f} rad/s")
    if tipo_circuito == "Subamortiguado":
        print(f"  ωd (frecuencia amortiguada) = {omega_d:.2f} rad/s")
        print(f"  Período = {2*np.pi/omega_d:.6f} s")
    print(f"Energía:")
    print(f"  Energía inicial del capacitor = {E_inicial:.4f} J")
    print(f"  Energía disipada en R = {E_R:.4f} J")
    print(f"  Energía neta en L = {E_L:.4f} J")
    print(f"  Energía neta en C = {E_C:.4f} J")
    print(f"  Energía total = {E_total:.4f} J")
    print(f"  Balance energético: {(E_total/E_inicial)*100:.2f}% de la energía inicial")
    
    plt.show()
    
    return {
        'tau': tau_efectivo,
        'energia_inicial': E_inicial,
        'energia_disipada': E_R,
        'energia_total': E_total
    }

def calcular_energia_circuito_rlc_interactivo():
    """
    Versión interactiva que permite modificar parámetros como masa y propiedades del inductor
    durante la simulación, recalculando los resultados en tiempo real.
    """
    # Parámetros iniciales tomados de main.py
    masa_inicial = m  # 0.01 kg - 10g
    largo_inicial = 0.04  # 4cm
    vueltas_inicial = 200  # vueltas
    diametro_alambre_inicial = 0.0005  # 0.5mm
    diametro_interno_inicial = 0.005  # 5mm
    V0_inicial = 400  # Voltios
    C_inicial = 470e-6  # Faradios

    # Crear figura y ejes
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 2, width_ratios=[3, 1])
    
    # Ejes para gráficos
    ax_corriente = fig.add_subplot(gs[0, 0])
    ax_tension = fig.add_subplot(gs[1, 0], sharex=ax_corriente)
    ax_potencia = fig.add_subplot(gs[2, 0], sharex=ax_corriente)
    ax_energia = fig.add_subplot(gs[3, 0], sharex=ax_corriente)
    
    # Ejes para panel de control
    ax_control = fig.add_subplot(gs[:, 1])
    ax_control.axis('off')  # Ocultar ejes
    
    # Posiciones para controles en el panel - Ajustado para evitar superposición
    control_y = 0.95  # Start higher
    control_altura = 0.04  # Smaller control height
    control_separacion = 0.1  # Increased separation
    slider_ancho = 0.25
    slider_x = 0.65
    etiqueta_x = 0.45
    
    # Función de actualización principal
    def actualizar_completo(event=None):
        # Limpiar completamente todos los ejes para forzar un redibujado total
        ax_corriente.clear()
        ax_tension.clear()
        ax_potencia.clear()
        ax_energia.clear()
        
        # Obtener valores actuales de los sliders
        masa = slider_masa.val
        largo = slider_largo.val
        N = int(slider_N.val)
        diametro_alambre = slider_diametro_alambre.val / 1000  # Convertir de mm a m
        diametro_interno = slider_diametro_interno.val / 1000  # Convertir de mm a m
        
        # Crear instancia del circuito con parámetros actualizados
        circuito = CircuitoRLC(
            V0=V0_inicial,  # Mantener constante
            C=C_inicial,    # Mantener constante
            rho_alambre=rho_cobre,
            largo=largo,
            N=N,
            diametro_interno=diametro_interno,
            diametro_alambre=diametro_alambre,
            verbose=False
        )
        
        # Obtener parámetros del circuito
        L = circuito.L_base
        R = circuito.R
        C = circuito.C
        V0 = circuito.V0
        
        # Calcular parámetros dinámicos
        tau_RL = L / R
        omega0 = 1 / np.sqrt(L * C)
        alpha = R / (2 * L)
        
        # Determinar tipo de circuito y constante de tiempo efectiva (tau)
        if alpha < omega0:
            omega_d = np.sqrt(omega0**2 - alpha**2)
            tau_efectivo = 1 / alpha
            tipo_circuito = "Subamortiguado"
        elif alpha == omega0:
            tau_efectivo = 1 / alpha
            tipo_circuito = "Críticamente amortiguado"
        else:
            tau_efectivo = 1 / alpha
            tipo_circuito = "Sobreamortiguado"
        
        # Configuración del tiempo para mostrar EXACTAMENTE 5*tau_efectivo
        t_max = 5 * tau_efectivo
        num_puntos = 5000
        
        # Generar array de tiempo basado en el nuevo tau
        tiempos = np.linspace(0, t_max, num_puntos)
        dt = tiempos[1] - tiempos[0]
        
        # Calcular corriente
        corrientes = np.array([circuito.corriente(t, L) for t in tiempos])
        
        # Derivada de corriente para tensión del inductor
        dI_dt = np.zeros_like(corrientes)
        dI_dt[1:-1] = (corrientes[2:] - corrientes[:-2]) / (2 * dt)
        dI_dt[0] = (corrientes[1] - corrientes[0]) / dt
        dI_dt[-1] = (corrientes[-1] - corrientes[-2]) / dt
        
        # Carga del capacitor
        carga_capacitor = np.zeros_like(corrientes)
        carga_capacitor[0] = C * V0
        
        for i in range(1, num_puntos):
            carga_capacitor[i] = carga_capacitor[i-1] - corrientes[i-1] * dt
        
        # Componentes de tensión
        V_R = R * corrientes
        V_L = L * dI_dt
        V_C = carga_capacitor / C
        
        # Potencias instantáneas - potencia transferida entre componentes
        P_R = V_R * corrientes      # Potencia disipada en R (siempre positiva)
        P_L = V_L * corrientes      # Potencia transferida al inductor (+ o -)
        P_C = V_C * corrientes      # Potencia entregada por el capacitor (- al principio)
        
        # Verificación: P_R + P_L + P_C debería ser cercano a cero (conservación de energía)
        P_suma = P_R + P_L - P_C

        
        # Energía almacenada en cada componente en cada instante
        # La energía almacenada no es la integral de la potencia, sino el estado energético instantáneo
        E_inicial = 0.5 * C * V0**2
        E_C_almacenada = 0.5 * C * V_C**2                   # Energía almacenada en el capacitor
        E_L_almacenada = 0.5 * L * corrientes**2            # Energía almacenada en el inductor
        
        # Calcular energía acumulada disipada en la resistencia (integral de P_R)
        E_R_acumulada = np.zeros_like(tiempos)
        for i in range(1, len(tiempos)):
            E_R_acumulada[i] = np.trapz(P_R[:i+1], tiempos[:i+1])
        
        # Calcular energía mecánica transferida a la masa
        # Inicializar objetos para los cálculos de energía mecánica
        histeresis = HisteresisHierroSilicio()  # Create instance directly
        
        # Posición inicial del núcleo (al inicio del solenoide)
        posicion_nucleo = np.full_like(tiempos, largo/2)  # Empezamos en el centro
        velocidad_nucleo = np.zeros_like(tiempos)
        aceleracion_nucleo = np.zeros_like(tiempos)
        energia_mecanica = np.zeros_like(tiempos)
        fuerza_mag = np.zeros_like(tiempos)
        
        # Cálculo de la dinámica del núcleo - modelo simplificado
        for i in range(1, len(tiempos)):
            # Calcular inductancia dinámica basada en la posición del núcleo
            L_dinamica = circuito.inductancia_dinamica(masa, posicion_nucleo[i-1], largo/2)
            
            # Obtener corriente con la inductancia dinámica
            I_actual = circuito.corriente(tiempos[i], L_dinamica)
            
            # Calcular fuerza magnética
            fuerza_mag[i] = circuito.calcular_fuerza(masa, posicion_nucleo[i-1], I_actual)
            
            # Calcular aceleración, velocidad y posición
            aceleracion_nucleo[i] = fuerza_mag[i] / masa
            
            # Integrar para obtener velocidad y posición (método de Euler simple)
            dt_local = tiempos[i] - tiempos[i-1]
            velocidad_nucleo[i] = velocidad_nucleo[i-1] + aceleracion_nucleo[i] * dt_local
            posicion_nucleo[i] = posicion_nucleo[i-1] + velocidad_nucleo[i] * dt_local
            
            # Calcular energía mecánica (cinética)
            energia_mecanica[i] = 0.5 * masa * velocidad_nucleo[i]**2
        
        # Calcular pérdidas por histéresis
        B_maximo = μ0 * N * np.max(corrientes) / largo
        H_maximo = B_maximo / μ0
        
        # Define area_ciclo_aproximada function inline if not available
        def area_ciclo_aproximada(H_max, masa):
            """Approximates the hysteresis loop area"""
            B_sat = histeresis.B_saturacion(masa)
            B_max = B_sat * np.tanh(H_max / B_sat)
            factor_reduccion = 1 - np.tanh(2)  # Factor para considerar forma del ciclo
            return 4 * H_max * B_max * factor_reduccion
        
        area_ciclo = area_ciclo_aproximada(H_maximo, masa)  # Área del ciclo de histéresis (J/m³)
        volumen_nucleo = masa / rho_material
        energia_histeresis = area_ciclo * volumen_nucleo  # Energía total perdida por histéresis
        
        # Crear array de pérdidas por histéresis (distribución proporcional a I²)
        # Las pérdidas por histéresis son proporcionales a la frecuencia y al área del ciclo
        I_squared = corrientes**2
        I_squared_norm = I_squared / np.sum(I_squared) if np.sum(I_squared) > 0 else np.zeros_like(I_squared)
        E_histeresis_acumulada = energia_histeresis * np.cumsum(I_squared_norm)
        
        # Energía total del sistema en cada instante, incluyendo mecánica y pérdidas por histéresis
        E_total_sistema = E_C_almacenada + E_L_almacenada + E_R_acumulada + energia_mecanica + E_histeresis_acumulada
        
        # Verificar conservación de energía con nuevos términos
        error_conservacion = 100 * abs(E_total_sistema[-1] - E_inicial) / E_inicial if E_inicial > 0 else 0
        
        # Actualizar estadísticas en el panel de control con nueva información
        texto_estadisticas = (
            f"Estadísticas del circuito:\n"
            f"------------------------\n"
            f"Tipo: {tipo_circuito}\n"
            f"R = {R:.4f} Ω\n"
            f"L = {L:.2e} H\n"
            f"C = {C:.2e} F\n"
            f"V0 = {V0:.1f} V\n"
            f"τ = {tau_efectivo:.6f} s\n"
            f"α = {alpha:.2f}\n"
            f"ω₀ = {omega0:.2f} rad/s\n"
        )
        
        if tipo_circuito == "Subamortiguado":
            texto_estadisticas += f"ωd = {omega_d:.2f} rad/s\n"
            texto_estadisticas += f"Período = {2*np.pi/omega_d:.6f} s\n"
        
        texto_estadisticas += (
            f"\nEnergía (Conservación):\n"
            f"Einicial (C) = {E_inicial:.4f} J\n"
            f"Efinal:\n"
            f"  - E disipada en R = {E_R_acumulada[-1]:.4f} J\n"
            f"  - E almacenada en L = {E_L_almacenada[-1]:.4f} J\n"
            f"  - E remanente en C = {E_C_almacenada[-1]:.4f} J\n"
            f"  - E mecánica núcleo = {energia_mecanica[-1]:.4f} J\n"
            f"  - E pérdidas histéresis = {E_histeresis_acumulada[-1]:.4f} J\n"
            f"  - E total sistema = {E_total_sistema[-1]:.4f} J\n"
            f"  - Error conservación = {error_conservacion:.6f}%\n"
            f"\nDistribución energía:\n"
            f"  - % en R: {(E_R_acumulada[-1]/E_inicial)*100:.1f}%\n"
            f"  - % en L: {(E_L_almacenada[-1]/E_inicial)*100:.1f}%\n"
            f"  - % en C: {(E_C_almacenada[-1]/E_inicial)*100:.1f}%\n"
            f"  - % en núcleo: {(energia_mecanica[-1]/E_inicial)*100:.1f}%\n"
            f"  - % histéresis: {(E_histeresis_acumulada[-1]/E_inicial)*100:.1f}%\n"
        )
        
        # Actualizar o crear nuevo texto de estadísticas
        for txt in ax_control.texts:
            if hasattr(txt, 'estadisticas_tag'):
                txt.remove()
                
        stats_text = ax_control.text(0.002, 0.01, texto_estadisticas,
                              transform=ax_control.transAxes, 
                              fontsize=8, family='monospace',
                              verticalalignment='bottom')
        stats_text.estadisticas_tag = True
        
        # Graficar corriente
        ax_corriente.plot(tiempos, corrientes, 'b-', linewidth=2, label="I(t)")
        ax_corriente.axvline(x=tau_efectivo, color='r', linestyle='--', label=f"τ = {tau_efectivo:.3f}s")
        for i in range(2, 6):
            ax_corriente.axvline(x=i*tau_efectivo, color='r', linestyle=':', alpha=0.5)
        ax_corriente.set_title(f'Corriente en el circuito {tipo_circuito}')
        ax_corriente.set_ylabel('Corriente (A)')
        ax_corriente.grid(True)
        ax_corriente.legend()
        ax_corriente.set_xlim(0, t_max)
        
        # Graficar tensiones
        ax_tension.plot(tiempos, V_R, 'r-', linewidth=2, label="V_R")
        ax_tension.plot(tiempos, V_L, 'g-', linewidth=2, label="V_L")
        ax_tension.plot(tiempos, V_C, 'b-', linewidth=2, label="V_C")
        ax_tension.axvline(x=tau_efectivo, color='k', linestyle='--')
        ax_tension.set_title('Tensiones en componentes del circuito')
        ax_tension.set_ylabel('Tensión (V)')
        ax_tension.grid(True)
        ax_tension.legend()
        ax_tension.set_xlim(0, t_max)
        
        # Graficar potencias
        ax_potencia.plot(tiempos, P_R, 'r-', linewidth=2, label="P_R")
        ax_potencia.plot(tiempos, P_L, 'g-', linewidth=2, label="P_L")
        ax_potencia.plot(tiempos, P_C, 'b-', linewidth=2, label="P_C")
        ax_potencia.plot(tiempos, P_suma, 'k--', linewidth=1, label="Suma (≈0)", alpha=0.5)
        ax_potencia.axvline(x=tau_efectivo, color='k', linestyle='--')
        ax_potencia.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax_potencia.set_title('Potencia instantánea (transferencia de energía)')
        ax_potencia.set_ylabel('Potencia (W)')
        ax_potencia.grid(True)
        ax_potencia.legend()
        ax_potencia.set_xlim(0, t_max)
        
        # Graficar energías almacenadas en cada componente, incluyendo energía mecánica
        ax_energia.plot(tiempos, E_R_acumulada, 'r-', linewidth=2, label="E disipada en R")
        ax_energia.plot(tiempos, E_L_almacenada, 'g-', linewidth=2, label="E almacenada en L")
        ax_energia.plot(tiempos, E_C_almacenada, 'b-', linewidth=2, label="E almacenada en C")
        ax_energia.plot(tiempos, energia_mecanica, 'm-', linewidth=2, label="E mecánica núcleo")
        ax_energia.plot(tiempos, E_histeresis_acumulada, 'y-', linewidth=2, label="E pérdidas histéresis")
        ax_energia.plot(tiempos, E_total_sistema, 'k-', linewidth=2, label="E total sistema")
        ax_energia.axvline(x=tau_efectivo, color='k', linestyle='--')
        ax_energia.axhline(y=E_inicial, color='m', linestyle='--', label=f"E_inicial = {E_inicial:.2f}J")
        ax_energia.set_title('Energía en el sistema (almacenada + disipada + mecánica)')
        ax_energia.set_xlabel('Tiempo (s)')
        ax_energia.set_ylabel('Energía (J)')
        ax_energia.grid(True)
        ax_energia.legend(fontsize=8)
        ax_energia.set_xlim(0, t_max)
        
        # Añadir un segundo gráfico para mostrar la posición y velocidad del núcleo
        # Solo si hay un segundo eje para la posición/velocidad
        if not hasattr(ax_energia, 'twin_axis'):
            ax_energia.twin_axis = ax_energia.twinx()
        else:
            ax_energia.twin_axis.clear()
            
        # Graficar posición del núcleo (normalizada)
        pos_line = ax_energia.twin_axis.plot(tiempos, posicion_nucleo, 'c--', alpha=0.7, 
                                            linewidth=1.5, label="Posición núcleo")
        ax_energia.twin_axis.set_ylabel('Posición (m)', color='c')
        ax_energia.twin_axis.tick_params(axis='y', labelcolor='c')
        
        # Añadir área del ciclo de histéresis a las constantes
        info_constantes = (
            f"Parámetros constantes:\n"
            f"μ₀ = {μ0:.1e} H/m\n"
            f"ρ_cobre = {rho_cobre:.2e} Ω·m\n"
            f"ρ_material = {rho_material} kg/m³\n"
            f"V₀ = {V0_inicial} V\n"
            f"C = {C_inicial*1e6:.1f} μF\n"
            f"Área ciclo histéresis = {area_ciclo:.2e} J/m³"
        )
        
        # Forzar redibujado completo
        fig.canvas.draw()
    
    # Crear etiquetas y sliders
    fig.text(etiqueta_x, control_y, "Masa (kg):", 
             transform=ax_control.transAxes, fontsize=10)
    ax_slider_masa = plt.axes([slider_x, control_y - 0.02, slider_ancho, control_altura], 
                              transform=ax_control.transAxes)
    slider_masa = Slider(ax_slider_masa, '', 0.001, 0.05, valinit=masa_inicial)
    control_y -= control_separacion
    
    fig.text(etiqueta_x, control_y, "Largo (m):", 
             transform=ax_control.transAxes, fontsize=10)
    ax_slider_largo = plt.axes([slider_x, control_y - 0.02, slider_ancho, control_altura], 
                               transform=ax_control.transAxes)
    slider_largo = Slider(ax_slider_largo, '', 0.01, 0.1, valinit=largo_inicial)
    control_y -= control_separacion
    
    fig.text(etiqueta_x, control_y, "Vueltas (N):", 
             transform=ax_control.transAxes, fontsize=10)
    ax_slider_N = plt.axes([slider_x, control_y - 0.02, slider_ancho, control_altura], 
                           transform=ax_control.transAxes)
    slider_N = Slider(ax_slider_N, '', 50, 500, valinit=vueltas_inicial, valfmt='%0.0f')
    control_y -= control_separacion
    
    fig.text(etiqueta_x, control_y, "Diám. alambre (mm):", 
             transform=ax_control.transAxes, fontsize=10)
    ax_slider_diametro_alambre = plt.axes([slider_x, control_y - 0.02, slider_ancho, control_altura], 
                                          transform=ax_control.transAxes)
    slider_diametro_alambre = Slider(ax_slider_diametro_alambre, '', 0.1, 2.0, 
                                     valinit=diametro_alambre_inicial*1000)
    control_y -= control_separacion
    
    fig.text(etiqueta_x, control_y, "Diám. interno (mm):", 
             transform=ax_control.transAxes, fontsize=10)
    ax_slider_diametro_interno = plt.axes([slider_x, control_y - 0.02, slider_ancho, control_altura], 
                                          transform=ax_control.transAxes)
    slider_diametro_interno = Slider(ax_slider_diametro_interno, '', 1.0, 20.0, 
                                     valinit=diametro_interno_inicial*1000)
    
    # Colocar el botón más a la derecha para evitar superposición
    ax_boton = plt.axes([slider_x + 0.2, control_y - 0.5, slider_ancho * 0.6, control_altura], 
                        transform=ax_control.transAxes)
    boton_actualizar = Button(ax_boton, 'Actualizar')
    boton_actualizar.on_clicked(actualizar_completo)
    
    control_y -= control_separacion*2
    
    info_constantes = (
        f"Parámetros constantes:\n"
        f"μ₀ = {μ0:.1e} H/m\n"
        f"ρ_cobre = {rho_cobre:.2e} Ω·m\n"
        f"ρ_material = {rho_material} kg/m³\n"
        f"V₀ = {V0_inicial} V\n"
        f"C = {C_inicial*1e6:.1f} μF"
    )
    constantes_text = ax_control.text(1, control_y, info_constantes,
                                      transform=ax_control.transAxes,
                                      fontsize=9, family='monospace')
    
    actualizar_completo()
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.show()
    

if __name__ == "__main__":
    print("=== ANÁLISIS ENERGÉTICO INTERACTIVO DEL CIRCUITO RLC ===")
    print("Use los controles laterales para modificar los parámetros del circuito")
    print("Los gráficos se actualizarán automáticamente")
    print()
    
    calcular_energia_circuito_rlc_interactivo()
