import tkinter as tk
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from histeresis import HisteresisHierroSilicio

# --- PARÁMETROS PRINCIPALES PARA OPTIMIZACIÓN ---
# Constante física global
μ0 = 4 * math.pi * 1e-7  # Permeabilidad del vacío (H/m)

rho_cobre = 1.68e-8      # Resistividad del cobre (Ohm * m) a 20°C

# Parámetro de física: masa del núcleo
m = 1e-2  # Masa del núcleo (kg) - PARÁMETRO PARA OPTIMIZACIÓN

# Parámetros del circuito que se pueden optimizar
# Estos valores se pasan al constructor de CircuitoRLC
# largo = 0.02       # Longitud del inductor (m) - PARÁMETRO PARA OPTIMIZACIÓN
# N = 200            # Número de vueltas - PARÁMETRO PARA OPTIMIZACIÓN  
# diametro_alambre = 0.001  # Diámetro del alambre (m) - PARÁMETRO PARA OPTIMIZACIÓN

# --- FIN PARÁMETROS PRINCIPALES PARA OPTIMIZACIÓN ---

# Density of iron-silicon (kg/m³)
rho_material = 7850  
dt = 0.01  # Base time step

escalas_tiempo = [1e-3, 1e-2, 1e-1, 1, 10, 100]

class CircuitoRLC:
    def __init__(self, V0=400, C=470e-6, rho_alambre=rho_cobre,
                 largo=0.04, N=200, diametro_interno=0.005, diametro_alambre=0.0001, verbose=True):
        """
        Inicializa el circuito RLC con los parámetros básicos, calculando los demás automáticamente.
        Parámetros fundamentales:
        - V0: Tensión inicial del capacitor (V)
        - C: Capacitancia del capacitor (F)
        - largo: Longitud física de la bobina (m)
        - N: Número de vueltas del inductor
        - diametro_interno: Diámetro interno del inductor (m)
        - diametro_alambre: Diámetro del alambre de cobre (m)
        - verbose: Si es True, imprime información sobre la configuración
        
        Parámetros calculados automáticamente:
        - capas: Número de capas necesarias para las N vueltas
        - diametro_medio: Diámetro medio de las vueltas (considerando todas las capas)
        - seccion_alambre: Área de la sección transversal del alambre (m²)
        - R: Resistencia del inductor (Ohm)
        - L_base: Inductancia base del inductor vacío (H)
        """
        self.V0 = V0
        self.C = C
        self.N = N
        self.largo = largo 
        self.diametro_interno = diametro_interno
        self.diametro_alambre = diametro_alambre
        self.rho_alambre = rho_alambre
        
        # Para compatibilidad con código que aún usa self.diametro
        self.diametro = self.diametro_interno  # Alias para compatibilidad con código existente
        
        # Calcular la sección del alambre a partir de su diámetro
        self.seccion_alambre = math.pi * (self.diametro_alambre/2)**2
        
        # Calcular número de vueltas por capa (considerando empaquetamiento óptimo)
        vueltas_por_capa = math.floor(self.largo / self.diametro_alambre)
        if vueltas_por_capa <= 0:
            raise ValueError("El largo del inductor es demasiado pequeño para el diámetro del alambre")
        
        # Calcular número de capas necesarias
        self.capas = math.ceil(self.N / vueltas_por_capa)
        
        # Calcular diámetro medio del bobinado (promedio de todas las capas)
        self.diametro_medio = self.diametro_interno + self.diametro_alambre * self.capas
        
        # Calcular longitud total del alambre
        longitud_media_vuelta = math.pi * self.diametro_medio
        longitud_total_alambre = self.N * longitud_media_vuelta
        
        # Calcular resistencia del inductor
        self.R = self.rho_alambre * longitud_total_alambre / self.seccion_alambre
        
        # Calcular inductancia base (bobina vacía)
        radio_interno = self.diametro_interno / 2
        Area_bobina = math.pi * radio_interno**2
        if self.largo > 0:
            self.L_base = μ0 * (self.N**2) * Area_bobina / self.largo
        else:
            self.L_base = 0
        
        # Instancia para cálculos de histéresis
        self.histeresis = HisteresisHierroSilicio()
        
        # Información para debugging y verificación - solo si verbose=True
        if verbose:
            print(f"[INFO] Configuración del inductor:")
            print(f"  - Largo: {self.largo*1000:.1f} mm")
            print(f"  - Vueltas (N): {self.N}")
            print(f"  - Diámetro interno: {self.diametro_interno*1000:.1f} mm")
            print(f"  - Diámetro alambre: {self.diametro_alambre*1000:.2f} mm")
            print(f"  - Capas calculadas: {self.capas}")
            print(f"  - Diámetro medio: {self.diametro_medio*1000:.1f} mm")
            print(f"  - Resistencia calculada: {self.R:.4f} Ohm")
            print(f"  - Inductancia base: {self.L_base:.6e} H")

    def inductancia_dinamica(self, masa, x, centro):
        """
        Calcula la inductancia dinámica en función de la posición x, considerando:
        - Adentro del inductor: inductancia con núcleo de hierro
        - Cerca: atenuación progresiva
        - Lejos: se asume núcleo de aire
        """
        r = self.diametro_interno / 2
        A = math.pi * r**2
        dx = abs(x - centro)

        # Definir regiones
        mitad = self.largo / 2
        margen_atenuacion = self.largo  # extensión del efecto fuera del núcleo
        limite_externo = mitad + margen_atenuacion

        # Campo magnético aproximado
        H_aprox = (self.N * self.V0) / (self.largo * self.R)
        B_aprox = self.histeresis.B(H_aprox, masa)
        mu_r = self.histeresis.mu_r(B_aprox, masa)
        μ_nucleo = μ0 * mu_r
        μ_aire = μ0

        if dx <= mitad:
            # Dentro del inductor → núcleo completamente acoplado
            μ = μ_nucleo
        elif dx <= limite_externo:
            # Zona de atenuación
            factor = 1 - ((dx - mitad) / margen_atenuacion)
            μ = μ_aire + factor * (μ_nucleo - μ_aire)
        else:
            # Fuera del alcance → núcleo de aire
            μ = μ_aire

        return μ * self.N**2 * A / self.largo

    def inductancia(self, masa, x, centro):
        """
        Calcula la inductancia en función de la posición x, la masa y el centro del inductor.
        """
        r = self.diametro_interno / 2  # Radio del núcleo
        A = math.pi * r**2  # Área de la sección transversal
        dx = abs(x - centro)  # Distancia al centro del inductor

        # Calcular el campo magnético aproximado H
        H_aprox = (self.N * self.V0) / (self.largo * self.R)  # H depende de la corriente y geometría
        B_aprox = self.histeresis.B(H_aprox, masa)  # Inducción magnética aproximada
        mu_r = self.histeresis.mu_r(B_aprox, masa)  # Permeabilidad relativa dinámica
        μ = μ0 * mu_r

        # Factor de acoplamiento basado en la posición del proyectil
        factor_posicion = max(0, 1 - (dx / (self.largo / 2))**2)  # Máximo en el centro, decae cuadráticamente

        # Inductancia dinámica
        if self.largo > 0:
            return factor_posicion * μ * self.N**2 * A / self.largo
        else:
            return 0

    def calcular_fuerza(self, masa, x, I):
        """
        Calcula la fuerza magnética sobre el proyectil.
        F(x, I) ≈ (1/2) * I^2 * dL/dx
        
        Utiliza una derivada más robusta con un paso mayor y un enfoque analítico-numérico
        para evitar errores numéricos que resultan en fuerza cero.
        """
        # Usar un paso mucho mayor para la derivada numérica
        dx = 0.01  # 1cm en lugar de fracciones de mm
        
        # Calcular inductancias en puntos más separados
        L_izq = self.inductancia_dinamica(masa, x - dx, self.largo/2)
        L_der = self.inductancia_dinamica(masa, x + dx, self.largo/2)
        
        # Derivada central más estable
        dL_dx = (L_der - L_izq) / (2 * dx)
        
        # Si la corriente es significativa pero la derivada está cerca de cero,
        # usar un valor mínimo de fuerza para evitar estancamiento
        if abs(dL_dx) < 1e-12 and abs(I) > 0.1:
            # Determinar dirección de la fuerza basada en posición
            signo = -1 if x > self.largo/2 else 1
            dL_dx = signo * 1e-10  # Valor mínimo con signo apropiado
        
        # Calcular la fuerza usando la fórmula F = 0.5 * I² * dL/dx
        fuerza = 0.5 * I * I * dL_dx
        
        return fuerza

    def corriente(self, t, L):
        """
        Calcula la corriente I(t) para la descarga RLC serie con V(0)=V0, I(0)=0.
        Utiliza la fórmula correcta para casos sub-, crítico y sobre-amortiguado.
        Usa la Resistencia calculada self.R.
        """
        if L <= 0 or self.C <= 0 or L * self.C <= 0:
            return 0.0
        R_usada = max(0, self.R) # Usar la R calculada (asegurando >=0)

        # Evitar división por cero si L es minúsculo
        if L < 1e-18: return 0.0

        alpha = R_usada / (2 * L)
        omega0_sq = 1 / (L * self.C)
        alpha_sq = alpha**2

        # Determinar caso de amortiguamiento con tolerancia numérica
        # Comparamos alpha^2 y omega0^2
        if abs(alpha_sq - omega0_sq) < 1e-9 * omega0_sq : # Críticamente Amortiguado
            amortiguamiento_tipo = 0
        elif alpha_sq > omega0_sq: # Sobreamortiguado
            amortiguamiento_tipo = 1
        else: # Subamortiguado
            amortiguamiento_tipo = -1

        try:
            if amortiguamiento_tipo == 0: # Caso Crítico
                if t < 0: return 0.0
                # Fórmula: I(t) = (V0 / L) * t * exp(-alpha * t)
                # Checkea si alpha o t son muy grandes -> exp underflow a 0
                if alpha * t > 700: return 0.0 # math.exp umbral approx
                corriente_t = (self.V0 / L) * t * math.exp(-alpha * t)

            elif amortiguamiento_tipo == 1: # Caso Sobreamortiguado
                if t < 0: return 0.0
                beta = math.sqrt(alpha_sq - omega0_sq)
                # Fórmula: I(t) = (V0 / (L * beta)) * exp(-alpha * t) * sinh(beta * t)
                if L * beta == 0: return 0.0 # Evitar división por cero
                # Checkear underflow/overflow potencial
                exp_arg = -alpha * t
                sinh_arg = beta * t
                if exp_arg < -700: return 0.0
                if abs(sinh_arg) > 700:
                     # sinh grande, usar exponenciales directamente para evitar overflow intermedio
                     # I(t) = V0 / (2*L*beta) * (exp(s1*t) - exp(s2*t)) donde s1=-alpha+beta, s2=-alpha-beta
                     s1 = -alpha + beta
                     s2 = -alpha - beta
                     if L*beta == 0: return 0.0
                     term1 = math.exp(s1 * t) if s1*t > -700 else 0.0
                     term2 = math.exp(s2 * t) if s2*t > -700 else 0.0
                     corriente_t = (self.V0 / (2*L*beta)) * (term1 - term2)
                else:
                     exp_term = math.exp(exp_arg)
                     sinh_term = math.sinh(sinh_arg)
                     corriente_t = (self.V0 / (L * beta)) * exp_term * sinh_term

            else: # Caso Subamortiguado
                if t < 0: return 0.0
                omega_d = math.sqrt(omega0_sq - alpha_sq)
                # Fórmula: I(t) = (V0 / (L * wd)) * exp(-alpha * t) * sin(wd * t)
                if L * omega_d == 0: return 0.0
                exp_arg = -alpha * t
                if exp_arg < -700: return 0.0 # exp underflow
                exp_term = math.exp(exp_arg)
                sin_term = math.sin(omega_d * t)
                corriente_t = (self.V0 / (L * omega_d)) * exp_term * sin_term

            # Manejar posibles NaN o Inf resultantes
            if math.isnan(corriente_t) or math.isinf(corriente_t):
                 return 0.0
            return corriente_t

        except (ValueError, OverflowError) as e:
            #print(f"Error numérico en cálculo de corriente: {e}")
            return 0.0

    def B_dinamico(self, x, masa, centro):
        """
        Devuelve una función que describe el campo magnético dinámico B(t) generado por la corriente en el circuito RLC.
        """
        # Calcular la inductancia dinámica en la posición x
        L = self.inductancia_dinamica(masa, x, centro)

        # Verificar que el largo del inductor sea válido
        if self.largo <= 0:
            raise ValueError("[ERROR] El largo del inductor es cero, no se puede calcular el campo magnético.")

        # Retornar una función que calcula B(t) en función del tiempo
        def B_t(t):
            I = self.corriente(t, L)  # Calcular la corriente en el tiempo t
            return (μ0 * self.N * I) / self.largo  # Calcular el campo magnético dinámico

        return B_t


class VisualizadorUI:
    """Clase encargada de la interfaz gráfica y visualización"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Campo Magnético con MAS")
        
        # Dimensiones del canvas
        self.ancho_canvas = 1200
        self.alto_canvas = 900
        
        # Crear canvas y frame de información
        self.canvas = tk.Canvas(root, width=self.ancho_canvas, height=self.alto_canvas, bg="white")
        self.canvas.pack(side=tk.LEFT)

        self.info_frame = tk.Frame(root)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Posiciones del inductor en canvas
        self.x0, self.x1 = 0, 200
        self.y0, self.y1 = 250, 350
        self.xc = (self.x0 + self.x1) / 2
        
        # Crear elementos básicos de UI
        self._crear_labels()
        self._crear_controles()
        self._crear_grafico()
        
        # Dibujar inductor
        self.canvas.create_rectangle(self.x0, self.y0, self.x1, self.y1, fill="lightblue", outline="black", width=2)
        
        # Variables para elementos visuales dinámicos
        self.punto = None
        self.flecha_fuerza = None
        self.trayectoria = []
        
    def _crear_labels(self):
        """Crea las etiquetas para mostrar valores dinámicos"""
        self.labels = {}
        for key in ["B local (T)", "Fuerza (N)", "Aceleración (m/s²)", "Velocidad (m/s)", "Posición (m)", "Distancia al centro (m)"]:
            label = tk.Label(self.info_frame, text=f"{key}: --", font=("Courier", 12))
            label.pack(anchor="w")
            self.labels[key] = label
            
        # Etiquetas adicionales
        self.label_B = tk.Label(self.info_frame, text="B (T): --", font=("Courier", 12))
        self.label_B.pack(anchor="w")

        self.label_H = tk.Label(self.info_frame, text="H (A/m): --", font=("Courier", 12))
        self.label_H.pack(anchor="w")

        self.label_tiempo_estimado = tk.Label(self.info_frame, text="Tiempo estimado paso campo: --", font=("Courier", 12))
        self.label_tiempo_estimado.pack(anchor="w")
        
        # Cronómetro
        self.cronometro_label = tk.Label(self.info_frame, text="Tiempo de simulación (s): 0.00", font=("Courier", 12))
        self.cronometro_label.pack(anchor="w")
        
        # Parámetros del sistema
        self.parametros_label = tk.Label(self.info_frame, text="Parámetros del sistema:", font=("Helvetica", 10, "bold"))
        self.parametros_label.pack(anchor="w", pady=(10, 0))

        self.label_masa = tk.Label(self.info_frame, text=f"Masa m (kg): {m:.1e}")
        self.label_masa.pack(anchor="w")

        self.label_densidad = tk.Label(self.info_frame, text=f"Densidad (kg/m³): {rho_material}")
        self.label_densidad.pack(anchor="w")
        
        self.label_dim_nucleo = tk.Label(self.info_frame, text="Núcleo: -- × --")
        self.label_dim_nucleo.pack(anchor="w")

        self.label_mu0 = tk.Label(self.info_frame, text=f"μ₀ (H/m): {μ0:.1e}")
        self.label_mu0.pack(anchor="w")

        self.label_mu_r = tk.Label(self.info_frame, text=f"μᵣ (adimensional): --")
        self.label_mu_r.pack(anchor="w")
        
        self.label_mu = tk.Label(self.info_frame, text=f"μ = μ₀·μᵣ (H/m): --")
        self.label_mu.pack(anchor="w")
        
        self.label_ep_inicial = tk.Label(self.info_frame, text="Energía potencial inicial capacitor (J): --")
        self.label_ep_inicial.pack(anchor="w")
        
        self.label_ek = tk.Label(self.info_frame, text="Energía cinética núcleo (J): --")
        self.label_ek.pack(anchor="w")
        
    def _crear_controles(self):
        """Crea los controles interactivos"""
        # Control de velocidad
        self.velocidad_simulacion = escalas_tiempo[3]
        self.slider_tiempo_label = tk.Label(self.info_frame, text=f"Velocidad de simulación (x): {self.velocidad_simulacion} s")
        self.slider_tiempo_label.pack(anchor="w")

        self.slider_tiempo = tk.Scale(self.info_frame, from_=0, to=len(escalas_tiempo) - 1, orient=tk.HORIZONTAL,
                                    label="Escala tiempo", tickinterval=1, resolution=1,
                                    showvalue=True)
        self.slider_tiempo.set(3)
        self.slider_tiempo.pack(anchor="w", fill=tk.X)
        
        # Botones
        self.boton_reiniciar = tk.Button(self.info_frame, text="Reiniciar simulación")
        self.boton_reiniciar.pack(pady=10)

        self.boton_disparar = tk.Button(self.info_frame, text="Disparar descarga")
        self.boton_disparar.pack(pady=5)
        
    def _crear_grafico(self):
        """Crea el gráfico para la trayectoria con mejor visualización"""
        # Increase figure size to make it wider (8,4 instead of 5,3)
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title("Posición vs Tiempo", fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Tiempo (s)", fontsize=10)
        self.ax.set_ylabel("Posición (m)", fontsize=10)
        
        # Create the main position line with improved styling
        self.linea_pos_tiempo, = self.ax.plot([], [], 'r-', label='Posición', lw=2)
        
        # Add reference line for inductor position
        self.ax.axhline(y=self.xc, color='blue', linestyle='--', alpha=0.5, label='Centro inductor')
        
        # Add grid for better readability
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add shaded region to represent inductor area
        self.inductor_area = self.ax.axhspan(self.x0, self.x1, alpha=0.2, color='lightblue', label='Inductor')
        
        # Improved legend with better positioning
        self.ax.legend(loc='upper right', framealpha=0.9)
        
        # Configure initial limits with better padding
        self.ax.set_xlim(0, 5)
        self.ax.set_ylim(self.x0 - 100, self.x1 + 100)
        
        # Create the figure canvas with better fit
        self.canvas_grafico = FigureCanvasTkAgg(self.fig, master=self.info_frame)
        self.canvas_grafico.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add subplot adjustment for better spacing
        plt.tight_layout()
    
    def dibujar_campo_estatico(self):
        """Dibuja las líneas de campo magnético"""
        # Limpiar líneas anteriores
        for item in self.canvas.find_withtag("campo"):
            self.canvas.delete(item)
        
        # Definir parámetros para la visualización
        centro_x = (self.x0 + self.x1) / 2
        centro_y = (self.y0 + self.y1) / 2
        ancho_rect = self.x1 - self.x0
        alto_rect = self.y1 - self.y0
        
        # Dibujar líneas internas
        y_step = max(1, int((self.y1 - self.y0) / 5))
        for y in range(int(self.y0) + y_step, int(self.y1), y_step):
            self.canvas.create_line(
                self.x0 + 10, y, self.x1 - 10, y, 
                fill="blue", arrow=tk.LAST, width=2, tags="campo"
            )
        
        # Usar valores aproximados para el dibujo
        margen_atenuacion = 40
        padding = 10
        
        # Crear efecto de sombra roja alrededor del inductor
        for i in range(3):
            width = 2 - i*0.5
            offset = i*3
            self.canvas.create_oval(
                self.x0 - padding - offset, 
                self.y0 - padding - offset,
                self.x1 + padding + offset, 
                self.y1 + padding + offset,
                outline="red", width=width, stipple="gray50",
                tags="campo"
            )
        
        # Crear elipses gradiente de rojo a amarillo con radios reducidos (1/4 del original)
        num_ellipses = 20
        # Dividir los radios por 4 para hacerlos más compactos
        max_radius_x = (ancho_rect + margen_atenuacion * 50) / 4  # Reducido a 1/4
        max_radius_y = (alto_rect + margen_atenuacion * 30) / 4   # Reducido a 1/4
        
        for i in range(1, num_ellipses + 1):
            ratio = i / num_ellipses
            radio_x = ancho_rect/2 + ratio * max_radius_x
            radio_y = alto_rect/2 + ratio * max_radius_y
            
            # Crear gradiente de color rojo a amarillo
            red = 255
            green = int(255 * ratio)
            blue = 0
            color = f"#{red:02x}{green:02x}{blue:02x}"
            
            # Crear elipse
            self.canvas.create_oval(
                centro_x - radio_x, centro_y - radio_y,
                centro_x + radio_x, centro_y + radio_y,
                outline=color, dash=(3, 3), tags="campo"
            )
        
        # Agregar leyenda
        legend_x = 50
        legend_y = 50
        self.canvas.create_rectangle(
            legend_x, legend_y,
            legend_x + 120, legend_y + 60,
            fill="white", outline="black", tags="campo"
        )
        self.canvas.create_line(
            legend_x + 10, legend_y + 15,
            legend_x + 30, legend_y + 15,
            fill="red", width=3, tags="campo"
        )
        self.canvas.create_line(
            legend_x + 10, legend_y + 35,
            legend_x + 30, legend_y + 35,
            fill="yellow", width=3, tags="campo"
        )
        self.canvas.create_text(
            legend_x + 75, legend_y + 15,
            text="Campo fuerte", fill="black", tags="campo"
        )
        self.canvas.create_text(
            legend_x + 75, legend_y + 35,
            text="Campo débil", fill="black", tags="campo"
        )
        
    def crear_proyectil(self, posicion_inicial, largo, altura):
        """Crea el proyectil en la posición inicial"""
        self.largo_nucleo_visual = largo
        self.altura_proyectil = altura
        self.y_centro_proyectil = (self.y0 + self.y1) / 2
        
        mitad_largo = self.largo_nucleo_visual / 2
        self.punto = self.canvas.create_rectangle(
            posicion_inicial - mitad_largo, self.y_centro_proyectil - self.altura_proyectil/2,
            posicion_inicial + mitad_largo, self.y_centro_proyectil + self.altura_proyectil/2,
            fill="red", outline="black", width=2
        )
        
        # Etiqueta para el proyectil
        self.etiqueta_nucleo = self.canvas.create_text(
            posicion_inicial, self.y_centro_proyectil - self.altura_proyectil/2 - 10,
            text="Núcleo Fe-Si", fill="black", font=("Arial", 10, "bold")
        )
        
    def actualizar_proyectil(self, posicion, fuerza):
        """Actualiza la posición visual del proyectil y la flecha de fuerza"""
        mitad_largo = self.largo_nucleo_visual / 2
        
        # Actualizar posición del rectángulo
        self.canvas.coords(self.punto, 
            posicion - mitad_largo, self.y_centro_proyectil - self.altura_proyectil/2,
            posicion + mitad_largo, self.y_centro_proyectil + self.altura_proyectil/2)
        
        # Actualizar posición de la etiqueta
        self.canvas.coords(
            self.etiqueta_nucleo,
            posicion, self.y_centro_proyectil - self.altura_proyectil/2 - 10
        )
        
        # Actualizar flecha de fuerza
        if self.flecha_fuerza:
            self.canvas.delete(self.flecha_fuerza)
            self.flecha_fuerza = None
            
        if fuerza != 0:
            escala_fuerza = 1e-10
            self.flecha_fuerza = self.canvas.create_line(
                posicion, self.y_centro_proyectil,
                posicion + fuerza * escala_fuerza, self.y_centro_proyectil,
                fill="red", arrow=tk.LAST, width=2
            )
            
    def actualizar_trayectoria(self, tiempo, posicion):
        """Actualiza el gráfico de trayectoria con mejor escalado dinámico"""
        self.trayectoria.append((tiempo, posicion))
        
        if len(self.trayectoria) > 1:
            # Extract time and position data
            tiempos = [p[0] for p in self.trayectoria]
            posiciones = [p[1] for p in self.trayectoria]
            
            # Update plot data
            self.linea_pos_tiempo.set_data(tiempos, posiciones)
            
            # Intelligent scaling:
            # For time axis: Always show a bit more than current time
            max_time = max(tiempos)
            self.ax.set_xlim(0, max_time * 1.2)
            
            # For position axis: Scale based on the min/max position with padding
            if max(posiciones) > self.x1 or min(posiciones) < self.x0:
                # If particle went outside inductor, use data-based limits
                pos_min = min(posiciones)
                pos_max = max(posiciones)
                pos_range = pos_max - pos_min
                
                # Add padding proportional to the range of motion
                padding = pos_range * 0.2 if pos_range > 0 else 100
                self.ax.set_ylim(pos_min - padding, pos_max + padding)
            else:
                # If still inside inductor area, keep fixed scale
                self.ax.set_ylim(self.x0 - 50, self.x1 + 50)
            
            # Update plot title with current velocity and position info
            if len(posiciones) > 1:
                current_pos = posiciones[-1]
                try:
                    current_vel = (posiciones[-1] - posiciones[-2]) / (tiempos[-1] - tiempos[-2])
                    self.ax.set_title(f"Posición vs Tiempo (Pos: {current_pos:.2f}m, Vel: {current_vel:.2f}m/s)")
                except (IndexError, ZeroDivisionError):
                    self.ax.set_title("Posición vs Tiempo")
            
            # Redraw the plot
            self.fig.canvas.draw_idle()  # More efficient than full redraw
    
    def actualizar_labels(self, datos):
        """Actualiza todas las etiquetas con los valores del estado físico"""
        self.labels["B local (T)"].config(text=f"B local (T): {datos['B_local']:.6e}")
        self.labels["Fuerza (N)"].config(text=f"Fuerza (N): {datos['fuerza']:.6e}")
        self.labels["Aceleración (m/s²)"].config(text=f"Aceleración (m/s²): {datos['aceleracion']:.6e}")
        self.labels["Velocidad (m/s)"].config(text=f"Velocidad (m/s): {datos['velocidad']:.6e}")
        self.labels["Posición (m)"].config(text=f"Posición (m): {datos['posicion']:.4f}")
        self.labels["Distancia al centro (m)"].config(text=f"Distancia al centro (m): {datos['distancia']:.4f}")
        
        self.label_B.config(text=f"B (T): {datos['B_base']:.6f}")
        self.label_H.config(text=f"H (A/m): {datos['H']:.6f}")
        
        self.label_mu_r.config(text=f"μᵣ (adimensional): {datos['mu_r']:.2f}")
        self.label_mu.config(text=f"μ = μ₀·μᵣ (H/m): {datos['mu']:.2e}")
        
        self.label_tiempo_estimado.config(text=f"Tiempo estimado paso campo: {datos['tiempo_estimado']:.3f} s")
        self.label_ek.config(text=f"Energía cinética núcleo (J): {datos['energia_cinetica']:.3e}")
        
    def actualizar_tiempo(self, tiempo):
        """Actualiza el tiempo mostrado en la interfaz"""
        self.cronometro_label.config(text=f"Tiempo de simulación (s): {tiempo:.2f}")
        
    def limpiar_trayectoria(self):
        """Limpia la trayectoria y reinicia el gráfico"""
        self.trayectoria.clear()
        self.linea_pos_tiempo.set_data([], [])
        
        # Reset plot appearance
        self.ax.set_title("Posición vs Tiempo")
        self.ax.set_xlim(0, 5)
        self.ax.set_ylim(self.x0 - 100, self.x1 + 100)
        
        # Clear any text annotations that might have been added
        for artist in self.ax.texts:
            artist.remove()
            
        self.canvas_grafico.draw()
        
        if self.flecha_fuerza:
            self.canvas.delete(self.flecha_fuerza)
            self.flecha_fuerza = None
    
    def limpiar_trayectoria(self):
        """Limpia la trayectoria y reinicia el gráfico"""
        self.trayectoria.clear()
        self.linea_pos_tiempo.set_data([], [])
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(self.x0 - 100, self.x1 + 100)
        self.canvas_grafico.draw()
        
        if self.flecha_fuerza:
            self.canvas.delete(self.flecha_fuerza)
            self.flecha_fuerza = None
            
    def configurar_dimensiones_nucleo(self, datos_nucleo):
        """Actualiza las etiquetas con las dimensiones del núcleo"""
        self.label_dim_nucleo.config(
            text=f"Núcleo: {datos_nucleo['diametro']*1000:.1f}mm × {datos_nucleo['largo']*1000:.1f}mm"
        )
        
    def configurar_energia_inicial(self, energia):
        """Muestra la energía potencial inicial del capacitor"""
        self.label_ep_inicial.config(text=f"Energía potencial inicial capacitor (J): {energia:.3e}")


class SimulacionFisica:
    """Clase encargada de los cálculos físicos de la simulación"""
    
    def __init__(self):
        # Inicializar objetos de cálculo
        self.histeresis = HisteresisHierroSilicio()
        self.circuito = CircuitoRLC()
        
        # Variables de estado
        self.descarga_activa = False
        self.t_descarga = 0
        self.B_t = None
        self.dt_physics = 0.001  # Fixed time step for physics calculations (1ms)
        
        # Configuración inicial
        self.posicion = 0
        self.velocidad = 0
        self.centro_inductor = 0
        
        # Cálculo de dimensiones del núcleo
        self.diametro_nucleo = self.circuito.diametro_interno
        self.area_nucleo = math.pi * (self.diametro_nucleo/2)**2
        self.largo_nucleo = m / (rho_material * self.area_nucleo)
        
        # Imprimir información
        print("\n=== INFORMACIÓN DEL NÚCLEO ===")
        print(f"Masa: {m:.2e} kg")
        print(f"Densidad: {rho_material} kg/m³")
        print(f"Diámetro: {self.diametro_nucleo*1000:.2f} mm")
        print(f"Largo calculado: {self.largo_nucleo*1000:.2f} mm")
        print(f"Volumen: {m/rho_material*1e6:.2f} cm³")
        print("=============================\n")
        
    def iniciar_descarga(self, posicion_inicial, centro):
        """Inicia una nueva descarga desde la posición indicada"""
        self.descarga_activa = True
        self.t_descarga = 0
        
        # Inicializar estado
        self.posicion = posicion_inicial
        self.velocidad = 0
        self.centro_inductor = centro
        
        # Inicializar función de campo magnético
        self.B_t = self.circuito.B_dinamico(self.posicion, m, centro)
        
    def detener_descarga(self):
        """Detiene la descarga actual"""
        self.descarga_activa = False
        
    def calcular_B_base(self, t):
        """Calcula el campo magnético base en el tiempo t"""
        if self.B_t is not None:
            return self.B_t(t)
        return 0.0
        
    def calcular_B_local(self, x):
        """Calcula el campo magnético local en la posición x"""
        if not self.descarga_activa:
            return 0, 0, 0
            
        # Calcular campo base
        B_base = self.calcular_B_base(self.t_descarga)
        
        # Calcular permeabilidad relativa
        mu_r = self.histeresis.mu_r(B_base, m)
        
        # Determinar atenuación por distancia
        centro = self.centro_inductor
        mitad = self.circuito.largo / 2
        x0_inductor = centro - mitad
        x1_inductor = centro + mitad
        
        if x0_inductor <= x <= x1_inductor:
            coef_atenuacion = 1.0
        else:
            dx = abs(x - centro)
            coef_atenuacion = 1 / (1 + (dx / mitad)**2)
            
        # Calcular campo local y campo H
        B_local = B_base * mu_r * coef_atenuacion
        H = B_base / μ0
        
        return B_local, B_base, H, mu_r
        
    def calcular_fuerza(self, x):
        """Calcula la fuerza magnética en la posición x"""
        if not self.descarga_activa or self.B_t is None:
            return 0.0
            
        # Obtener corriente real I(t) usando inductancia dinámica
        L = self.circuito.inductancia_dinamica(m, x, self.centro_inductor)
        I = self.circuito.corriente(self.t_descarga, L)
        
        # Calcular fuerza
        F = self.circuito.calcular_fuerza(m, x, I)
        return F
        
    def paso_simulacion(self, dt):
        """Ejecuta un paso de simulación física con intervalo dt"""
        if not self.descarga_activa:
            return self._crear_estado_inactivo()
            
        # Actualizar campo B(t)
        self.B_t = self.circuito.B_dinamico(self.posicion, m, self.centro_inductor)
        
        # Calcular campos y fuerzas
        B_local, B_base, H, mu_r = self.calcular_B_local(self.posicion)
        
        # Aplicar fuerzas después de un tiempo inicial de estabilización
        tau = self.circuito.L_base / self.circuito.R
        tiempo_inicial_estable = tau/100
        
        if self.t_descarga > tiempo_inicial_estable:
            fuerza = self.calcular_fuerza(self.posicion)
            aceleracion = fuerza / m
        else:
            fuerza = 0
            aceleracion = 0
            
        # Actualizar velocidad y posición
        self.velocidad += aceleracion * dt
        self.posicion += self.velocidad * dt
        
        # Incrementar tiempo
        self.t_descarga += dt
        
        # Calcular valores adicionales
        distancia_centro = self.posicion - self.centro_inductor
        energia_cinetica = 0.5 * m * self.velocidad**2
        tiempo_estimado = abs(distancia_centro) / abs(self.velocidad) if self.velocidad != 0 else float('inf')
        
        # Crear y devolver estado actual
        return {
            'posicion': self.posicion,
            'velocidad': self.velocidad,
            'aceleracion': aceleracion,
            'fuerza': fuerza,
            'B_local': B_local,
            'B_base': B_base,
            'H': H,
            'mu_r': mu_r,
            'mu': μ0 * mu_r,
            'distancia': distancia_centro,
            'energia_cinetica': energia_cinetica,
            'tiempo_estimado': tiempo_estimado,
            't_descarga': self.t_descarga
        }
        
    def _crear_estado_inactivo(self):
        """Crea un estado con valores predeterminados para cuando no hay descarga activa"""
        return {
            'posicion': self.posicion,
            'velocidad': 0,
            'aceleracion': 0,
            'fuerza': 0,
            'B_local': 0,
            'B_base': 0,
            'H': 0,
            'mu_r': 1,
            'mu': μ0,
            'distancia': self.posicion - self.centro_inductor,
            'energia_cinetica': 0,
            'tiempo_estimado': float('inf'),
            't_descarga': 0
        }
        
    def obtener_info_nucleo(self):
        """Devuelve información sobre el núcleo para mostrar en la UI"""
        return {
            'diametro': self.diametro_nucleo,
            'largo': self.largo_nucleo,
            'masa': m,
            'densidad': rho_material
        }
        
    def obtener_energia_inicial(self):
        """Devuelve la energía potencial inicial del capacitor"""
        return 0.5 * self.circuito.C * self.circuito.V0**2


class AplicacionSimulacion(VisualizadorUI, SimulacionFisica):
    """Clase principal que coordina la interfaz y la física de la simulación"""
    
    def __init__(self, root):
        # Inicializar componentes
        VisualizadorUI.__init__(self, root)
        SimulacionFisica.__init__(self)
        
        # Asociar eventos UI con métodos
        self.slider_tiempo.config(command=self.cambiar_velocidad)
        self.boton_disparar.config(command=self.disparar)
        self.boton_reiniciar.config(command=self.reiniciar)
        
        # Variables de control de simulación
        self.tiempo_simulacion = 0
        self.acumulador_tiempo = 0
        
        # Establecer dimensiones del núcleo en la UI
        datos_nucleo = self.obtener_info_nucleo()
        self.configurar_dimensiones_nucleo(datos_nucleo)
        
        # Mostrar energía inicial
        self.configurar_energia_inicial(self.obtener_energia_inicial())
        
        # Dimensiones visuales
        self.largo_nucleo_visual = max(10, self.largo_nucleo * 1000)
        self.altura_proyectil = min(40, (self.y1 - self.y0) * 0.5)
        
        # Crear proyectil
        self.crear_proyectil(self.x0, self.largo_nucleo_visual, self.altura_proyectil)
        
        # Dibujar campo magnético estático
        self.dibujar_campo_estatico()
        
        # Iniciar animación
        self.animar()
    
    def cambiar_velocidad(self, val):
        """Cambia la velocidad de simulación"""
        indice = int(val)
        self.velocidad_simulacion = escalas_tiempo[indice]
        self.slider_tiempo_label.config(text=f"Velocidad de simulación (x): {self.velocidad_simulacion} s")
    
    def disparar(self):
        """Inicia la descarga del circuito"""
        # Iniciar descarga en el modelo físico
        self.iniciar_descarga(self.x0, self.xc)
        
        # Limpiar visualización
        self.limpiar_trayectoria()
        
        # Reiniciar tiempo de simulación
        self.tiempo_simulacion = 0
        self.acumulador_tiempo = 0
        
        # Actualizar UI
        self.actualizar_tiempo(0)
    
    def reiniciar(self):
        """Reinicia la simulación a su estado inicial"""
        # Detener descarga
        self.detener_descarga()
        
        # Restablecer posición inicial
        self.posicion = self.x0
        self.velocidad = 0
        
        # Actualizar proyectil
        self.actualizar_proyectil(self.x0, 0)
        
        # Limpiar visualización
        self.limpiar_trayectoria()
        
        # Reiniciar tiempo
        self.tiempo_simulacion = 0
        self.acumulador_tiempo = 0
        self.actualizar_tiempo(0)
        
        # Mostrar valores iniciales en la UI
        estado_inicial = self._crear_estado_inactivo()
        self.actualizar_labels(estado_inicial)
    
    def animar(self):
        """Bucle principal de animación"""
        if self.descarga_activa:
            # Calcular tiempo físico a simular
            dt_visual = dt * self.velocidad_simulacion
            self.acumulador_tiempo += dt_visual
            
            # Ejecutar pasos de física con paso fijo
            num_pasos = 0
            while self.acumulador_tiempo >= self.dt_physics and num_pasos < 100:
                # Ejecutar un paso de simulación
                estado = self.paso_simulacion(self.dt_physics)
                
                # Reducir acumulador
                self.acumulador_tiempo -= self.dt_physics
                num_pasos += 1
            
            # Si se ejecutaron pasos, actualizar visualización
            if num_pasos > 0:
                # Actualizar proyectil
                self.actualizar_proyectil(estado['posicion'], estado['fuerza'])
                
                # Actualizar etiquetas
                self.actualizar_labels(estado)
                
                # Actualizar gráfico
                self.actualizar_trayectoria(self.tiempo_simulacion, estado['posicion'])
        
        # Actualizar tiempo
        self.tiempo_simulacion += dt * self.velocidad_simulacion
        self.actualizar_tiempo(self.tiempo_simulacion)
        
        # Programar próximo frame
        self.root.after(int(dt * 1000), self.animar)


if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionSimulacion(root)
    root.mainloop()