import numpy as np
import cupy as cp
import safe_math as sm
#*Constantes*#
# Definición de constantes para la simulación

capacitance=np.float32(2000e-6) #capacitancia
initial_voltage=np.float32(400) #tensión inicial
initial_current=np.float32(0) #corriente inicial
copper_rho=np.float32(1.7e-8) #resistividad del cobre
core_rho=np.float32(0.5e-6) #resistividad del hierro-silicio
ABH= np.float32(250) #área de lazo histerético
MagneticSaturationFluxDensity=np.float32(1.5) #inducción magnética de saturación hierro-silicio
copper_density = np.float32(8960) #densidad del cobre
core_density = np.float32(7650) #densidad del hierro-silicio
core_mass = np.float32(0.003) #masa del núcleo
core_relative_permeability = np.float32(5000) #permeabilidad relativa del hierro-silicio
vacuum_permeability = np.float32(4*np.pi*1e-7) #permeabilidad del vacío
dt=np.float32(1e-4) #intervalo de tiempo

#*Constantes paramétricas*#
# Definición de constantes para la simulación

# n_v = cp.arange(10, 801, 1, dtype=cp.float32)             #número de vueltas
# d_c= cp.arange(0.0001, 0.0015, 0.0001, dtype=cp.float32)  #diámetro del cobre
# d_n = cp.arange(0.004, 0.01, 0.001, dtype=cp.float32)     #diámetro del núcleo
# l_b = cp.arange(0.01, 0.4, 0.01, dtype=cp.float32)        #longitud de la bobina

number_of_turns=np.float32(100) #número de vueltas
copper_diameter=np.float32(0.001) #diámetro del cobre
core_diameter=np.float32(0.005) #diámetro del núcleo
coil_length=np.float32(0.1) #longitud de la bobina

#*Constantes fijas*#
# Definición de constantes para la simulación
core_volume=core_mass/core_density #volumen del núcleo
copper_area=np.float32(cp.pi*(copper_diameter/2)**2) #área del cobre
core_area=np.float32(cp.pi*(core_diameter/2)**2) #área del núcleo
number_of_layers=-((-number_of_turns*copper_diameter)//coil_length) #número de capas
print(f"número de capas: {number_of_layers}")
minimum_coil_diameter=core_diameter #diámetro mínimo bobina
turns_per_layer=(coil_length//(copper_diameter)) #numero de vueltas de cobre por capa
print(f"vueltas de cobre por capa: {turns_per_layer}")
turns_in_full_layers = (number_of_layers - 1) * turns_per_layer
turns_in_last_layer = number_of_turns - turns_in_full_layers #vueltas de cobre en la capa final
print(f"vueltas de cobre en la capa final: {turns_in_last_layer}")

# Cálculo de la longitud del cobre
copper_length = np.float32(0)
average_coil_diameter = np.float32(0) #diámetro medio de la bobina
for i in range(int(number_of_layers)):
    diametro_capa = core_diameter + 2 * copper_diameter * i
    vueltas_en_capa = turns_per_layer if i < number_of_layers-1 else turns_in_last_layer
    average_coil_diameter += core_diameter+2*copper_diameter*i if i < number_of_layers-1 or number_of_layers==1 else 2 * copper_diameter * (turns_in_last_layer/turns_per_layer)*i
    print(f"vueltas en capa {i}: {vueltas_en_capa}")
    print(f"diámetro de la capa {i}: {diametro_capa:.4f} m")
    copper_length += vueltas_en_capa * np.pi * diametro_capa
print(f"longitud del cobre: {copper_length:.4f} m")
print(f"diametro medio del inductor: {average_coil_diameter:.4f} m")
average_coil_area=np.float32(cp.pi*(average_coil_diameter/2)**2) #área media de la bobina
copper_mass=np.float32(copper_area*copper_length*copper_density) #masa del cobre
core_length=np.float32(core_mass/(core_density*core_area))#longitud del núcleo
KE=np.float32((core_diameter)**2/core_rho) #constante de la ley de ohm
copper_resistance=np.float32((copper_rho*copper_length)/copper_area) #resistencia del cobre
initial_inductance = np.float32((vacuum_permeability * core_relative_permeability * number_of_turns**2 * average_coil_area) / coil_length) #inductancia inicial
coil_midpoint = np.float32(coil_length / 2)  # Centro del inductor

#*Ecuaciones diferenciales*#
# Definición de las ecuaciones diferenciales para el circuito RLC
alpha=np.float32(copper_resistance/(2*initial_inductance)) #constante de la ecuación diferencial
omega_0=np.float32(1/(sm.safe_sqrt(initial_inductance*capacitance))) #frecuencia angular
discriminante=np.float32(alpha**2-omega_0**2) #discriminante de la ecuación diferencial

def subamortiguado():
    #oscila hasta que se detiene, por lo que la corriente y tensión son funciones sinusoidales amortiguadas
    """Calcula la velocidad y corriente para el caso subamortiguado."""
    omega_d=sm.safe_sqrt(omega_0**2-alpha**2) #frecuencia angular amortiguada
    phi=sm.safe_arctan(-omega_d/alpha) #fase de la tensión
    #* i(t)= i_0*e^(-alpha*t)*sin(omega_d*t)
    i=lambda t: initial_current*sm.safe_exp(-alpha*t)*sm.safe_sin(omega_d*t) #corriente
    #* v(t)=v_0*e^(-alpha*t)*((cos(omega_d*t+phi))**2)/2
    v=lambda t: initial_voltage*sm.safe_exp(-alpha*t)*((sm.safe_cos(omega_d*t+phi))**2)/2 #tensión 
    return i,v

def criticamenteAmortiguado():
    #La corriente y tensión decaen exponencialmente sin oscilaciones.
    """Calcula la velocidad y corriente para el caso críticamente amortiguado."""
    s1=-alpha-sm.safe_sqrt(alpha**2 - omega_0) #raíz 1 
    s2=-alpha+sm.safe_sqrt(alpha**2 - omega_0) #raíz 2
    A=sm.safe_divide(-initial_voltage,initial_inductance*(s1-s2)) #constante A
    B=sm.safe_divide(initial_voltage,initial_inductance*(s1-s2)) #constante B
    i=lambda t: (A+B*t)*sm.safe_exp(-alpha*t) #corriente
    return i,initial_voltage

def sobreamortiguado():
    #La corriente y tensión decaen sin oscilaciones en un periodo largo.
    """Calcula la corriente y tensión para el caso sobreamortiguado."""
    s1 = -alpha + sm.safe_sqrt(alpha**2 - omega_0**2)  # raíz 1
    s2 = -alpha - sm.safe_sqrt(alpha**2 - omega_0**2)  # raíz 2
    # Constantes A y B determinadas por condiciones iniciales
    denom = s1 - s2
    if denom == 0:
        denom = 1e-12  # evitar división por cero
    A = (initial_current - (s2 * initial_voltage * capacitance) / initial_inductance) / denom
    B = (-(initial_current - (s1 * initial_voltage * capacitance) / initial_inductance)) / denom
    i = lambda t: A * sm.safe_exp(s1 * t) + B * sm.safe_exp(s2 * t)
    v = lambda t: initial_voltage * sm.safe_exp(-alpha * t)
    return i, v


match discriminante:
    case _ if  (discriminante > -sm.epsilon) and (discriminante < sm.epsilon): #casi cero
     current_over_time,voltage_over_time=subamortiguado()
     print("subamortiguado")
    case _ if discriminante > 0:
     current_over_time,voltage_over_time=criticamenteAmortiguado()
     print("criticamente amortiguado")
    case _:
     current_over_time,voltage_over_time=sobreamortiguado()
     print("sobreamortiguado")
total_simulation_duration=np.float32(6*initial_inductance/copper_resistance) #tiempo de simulación

#*ciclo de simulación*# 
# Inicialización para cálculo discreto
inicial_position = np.float32(0)  # Posición inicial del núcleo
position = np.float32(1e-6)  # Posición del núcleo
B_anterior = np.float32(1e-6)  # Flujo magnético anterior
W_termico = np.float32(0)  # Trabajo térmico
W_foucault = np.float32(0)  # Trabajo de Foucault
W_histeretico = np.float32(0)  # Trabajo de histéresis
W_magnetico_anterior = np.float32(0) # Trabajo magnético inicial
W_magnético = np.float32(0)  # Trabajo magnético
W_cinetico = np.float32(0)  # Trabajo cinético
speed = np.float32(0)  # Velocidad del núcleo
w_sat = np.float32(0)  # Trabajo magnético de saturación

w_sat = 0.5 * core_volume * MagneticSaturationFluxDensity**2 / (vacuum_permeability * core_relative_permeability) #!revisar si es correcto
print(f"Trabajo magnético de saturación: {w_sat:.6f} J")
for t in np.arange(np.float32(0), total_simulation_duration, dt, dtype=np.float32):    
    i_t = current_over_time(t) # Corriente instantánea
    v_t = voltage_over_time(t) if callable(voltage_over_time) else initial_voltage  # Tensión instantánea

    W_electrico = np.float32(v_t**2 * capacitance / 2)  # Trabajo eléctrico del capacitor
    W_magnético = np.float32(min(W_electrico - W_termico - W_foucault - W_histeretico - W_cinetico, w_sat)) # Trabajo magnético 
    delta_position = np.float32(position - inicial_position)  # Cambio en la posición del núcleo
    # Protección robusta para evitar divisiones por valores demasiado pequeños
    min_delta = 1e-32
    if abs(delta_position) < min_delta:
        magnetic_force = np.float32(0)
    else:
        magnetic_force = np.float32((W_magnético - W_magnetico_anterior) / (delta_position))
    if position >= coil_midpoint:
        magnetic_force *= np.float32(-1) # Invertir la fuerza magnética si el núcleo está más allá del centro de la bobina
    
    W_magnetico_anterior = W_magnético  #? Actualizar el trabajo magnético inicial para el siguiente ciclo
    aceleration = np.float32(magnetic_force / core_mass)  # Actualización de la aceleración del núcleo
    speed += np.float32(aceleration * dt)  # Actualización de la velocidad
    position += np.float32(speed * dt)  # Actualización de la posición del núcleo
    inicial_position=position  #? Actualización de la posición previa del núcleo para el siguiente ciclo
    # Inductance = initial_inductance  # Inductancia inicial
    Inductance = (number_of_turns * vacuum_permeability * copper_area) / (coil_length - position + position / core_relative_permeability)

    # Cálculo del flujo magnético
    B = np.float32(vacuum_permeability * core_relative_permeability * (number_of_turns / coil_length) * i_t)  # flujo magnético
    dB_dt = np.float32((B - B_anterior) / dt) if t > 0 else np.float32(0)
    # Saturación magnética
    B_eff = np.float32(np.clip(B, -MagneticSaturationFluxDensity, MagneticSaturationFluxDensity))
    B_anterior_eff = np.float32(np.clip(B_anterior, -MagneticSaturationFluxDensity, MagneticSaturationFluxDensity))
    B_anterior = B #? Actualizar B_anterior para el siguiente ciclo
    #trabajos de pérdida recalculados
    W_termico += np.float32(copper_resistance * i_t ** 2 * dt)  # Joule
    W_foucault += np.float32(KE * dB_dt ** 2 * core_volume * dt)  # Foucault
    W_histeretico += np.float32(core_volume * ABH * abs(B_eff - B_anterior_eff) * dt)
    W_cinetico= np.float32(speed**2 * core_mass / 2)  # Trabajo cinético del núcleo
    
def prints():

    print(f"Tiempo: {t:.4f} s, Corriente: {i_t:.4f} A, Tensión: {v_t:.4f} V, Velocidad: {speed:.4f} m/s, Posición: {position:.4f} m, Inductancia: {Inductance:.4f} H")
    print("velocidad = ",speed, "m/s")

    print("\n--- PARÁMETROS CONSTRUCTIVOS DE LA BOBINA ---")
    print(f"N° de vueltas: {number_of_turns}")
    print(f"Diámetro del cobre: {copper_diameter:.6f} m")
    print(f"Diámetro del núcleo: {core_diameter:.6f} m")
    print(f"Longitud de la bobina: {coil_length:.4f} m")
    print(f"Longitud total de cobre: {copper_length:.4f} m")
    print(f"Masa del cobre: {copper_mass:.4f} kg")
    print(f"Masa del núcleo: {core_mass:.4f} kg")
    print(f"Resistencia del cobre: {copper_resistance:.4f} Ω")
    print(f"Inductancia inicial: {initial_inductance:.6f} H")

    # Corriente pico
    print("\n--- CORRIENTE PICO ---")
    # Buscar corriente máxima durante la simulación
    tiempos = np.arange(np.float32(0), total_simulation_duration, dt, dtype=np.float32)
    I_vals = np.array([np.float32(current_over_time(t)) for t in tiempos], dtype=np.float32)
    I_peak = np.max(np.abs(I_vals))
    print(f"Corriente pico: {I_peak:.4f} A")

    # Trabajos y porcentajes
    W_inicial = np.float32(initial_voltage**2 * capacitance / 2)
    print("\n--- BALANCE DE ENERGÍA ---")
    print(f"Trabajo inicial (condensador): {W_inicial:.6f} J")
    print(f"Trabajo disipado por Joule: {W_termico:.6f} J ({100*W_termico/W_inicial:.2f}%)")
    print(f"Trabajo disipado por Foucault: {W_foucault:.6f} J ({100*W_foucault/W_inicial:.2f}%)")
    print(f"Trabajo disipado por histéresis: {W_histeretico:.6f} J ({100*W_histeretico/W_inicial:.2f}%)")
    print(f"Trabajo cinético final: {W_cinetico:.6f} J ({100*W_cinetico/W_inicial:.2f}%)")

    # Eficiencia
    print("\n--- EFICIENCIA ---")
    print(f"Eficiencia energética: {100*W_cinetico/W_inicial:.2f}%")
    # ---
    # Todas las variables y arrays críticos de la simulación se han convertido a float32 para máxima resolución y estabilidad numérica.
    # ---

prints()