import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, make_interp_spline
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.colors import LogNorm
from matplotlib import cm 

def cargar_datos():
    """Carga los datos desde CSV"""
    print("Cargando datos de simulación...")
    df = pd.read_csv("resultados_simulacion.csv")
    return (df["v_final"].values, df["N"].values, df["d_hilo"].values, 
            df["d_nucleo"].values, df["l_nucleo"].values, df["l_bobina"].values,
            df["R"].values, df["tipo_amort"].values.astype(int))

def superficie_maxima(x, y, z, res=100):
    """Crea una superficie interpolada de los máximos"""
    puntos = np.column_stack((x, y))
    df_max = {}
    for xi, yi, zi in zip(x, y, z):
        key = (xi, yi)
        if key not in df_max or zi > df_max[key]:
            df_max[key] = zi
    puntos_max = np.array(list(df_max.keys()))
    z_max = np.array(list(df_max.values()))
    xi = np.linspace(x.min(), x.max(), res)
    yi = np.linspace(y.min(), y.max(), res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(puntos_max, z_max, (Xi, Yi), method='cubic')
    return Xi, Yi, Zi
    
def envolvente_suave(x, y, puntos=200):
    """Genera una envolvente suavizada"""
    x = np.array(x)
    y = np.array(y)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    x_smooth = np.linspace(x.min(), x.max(), puntos)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

class VisualizadorGraficos:
    def __init__(self):
        # Cargar datos
        self.v_finales_np, self.vueltas_np, self.d_hilo_np, self.d_nucleo_np, \
        self.l_nucleo_np, self.l_bobina_np, self.resistencia_np, self.tipo_amort_np = cargar_datos()
        
        # Configurar figura principal
        self.fig = plt.figure(figsize=(14, 9))
        self.indice_actual = 0
        self.titulos = [
            "Velocidad máxima vs Vueltas",
            "Velocidad vs Resistencia por tipo de amortiguamiento",
            "Superficie: Velocidad vs Vueltas y diámetro de hilo",
            "Velocidad máxima vs diámetro del núcleo"
        ]
        
        # Agregar botones de navegación
        self.agregar_controles()
        
        # Dibujar el primer gráfico
        self.actualizar_grafico()
        
    def agregar_controles(self):
        """Agrega botones de navegación y texto informativo"""
        self.ax_prev = self.fig.add_axes([0.2, 0.01, 0.1, 0.05])
        self.ax_next = self.fig.add_axes([0.7, 0.01, 0.1, 0.05])
        
        self.btn_prev = Button(self.ax_prev, 'Anterior')
        self.btn_next = Button(self.ax_next, 'Siguiente')
        
        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)
        
        self.info_text = self.fig.text(0.5, 0.01, "", 
                                     ha='center', va='center', fontsize=12)
        
        # Conectar eventos de teclado
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def on_prev(self, event):
        """Callback para botón anterior"""
        self.indice_actual = (self.indice_actual - 1) % len(self.titulos)
        self.actualizar_grafico()
    
    def on_next(self, event):
        """Callback para botón siguiente"""
        self.indice_actual = (self.indice_actual + 1) % len(self.titulos)
        self.actualizar_grafico()
    
    def on_key(self, event):
        """Callback para eventos de teclado"""
        if event.key == 'left':
            self.indice_actual = (self.indice_actual - 1) % len(self.titulos)
        elif event.key == 'right':
            self.indice_actual = (self.indice_actual + 1) % len(self.titulos)
        elif event.key in ['1', '2', '3', '4']:
            self.indice_actual = int(event.key) - 1
        elif event.key == 'q':
            plt.close('all')
            return
            
        self.actualizar_grafico()
    
    def actualizar_grafico(self):
        """Actualiza el contenido del gráfico según el índice actual"""
        # Limpiar figura pero mantener los ejes de navegación
        for ax in self.fig.axes:
            if ax != self.ax_prev and ax != self.ax_next:
                self.fig.delaxes(ax)
        
        # Actualizar texto informativo
        texto = f"Gráfico {self.indice_actual+1}/{len(self.titulos)}: {self.titulos[self.indice_actual]} (← → para navegar, 1-4 selección directa, q para salir)"
        self.info_text.set_text(texto)
        
        # Crear el gráfico actual
        if self.indice_actual == 0:
            self.crear_fig_vueltas()
        elif self.indice_actual == 1:
            self.crear_fig_resistencia()
        elif self.indice_actual == 2:
            self.crear_fig_superficie()
        elif self.indice_actual == 3:
            self.crear_fig_nucleo()
        
        # Ajustar layout y mostrar
        plt.tight_layout(rect=[0, 0.07, 1, 0.95])  # Dejar espacio para botones
        self.fig.canvas.draw_idle()
        
        # Maximizar ventana
        try:
            self.fig.canvas.manager.window.showMaximized()  # Para Windows
        except:
            pass
    
    def crear_fig_vueltas(self):
        """Gráfico 1: Velocidad máxima vs Vueltas"""
        ax = self.fig.add_subplot(111)
        
        vueltas_unique = np.unique(self.vueltas_np)
        max_v_por_vuelta = [self.v_finales_np[self.vueltas_np == v].max() for v in vueltas_unique]
        
        # Aplicar suavizado
        try:
            x_smooth, y_smooth = envolvente_suave(vueltas_unique, max_v_por_vuelta)
            ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label="Envolvente suavizada")
        except:
            pass  # Si falla el suavizado

    
    def crear_fig_resistencia(self):
        """Gráfico 2: Velocidad vs Resistencia por tipo de amortiguamiento"""
        ax = self.fig.add_subplot(111)
        
        categorias = ['Subamortiguado', 'Crítico', 'Sobreamortiguado']
        colores = ['blue', 'green', 'red']
        
        # Procesar cada tipo de amortiguamiento por separado
        for tipo_amort, color, nombre in zip([0, 1, 2], colores, categorias):
            # Filtrar solo este tipo
            mask = self.tipo_amort_np == tipo_amort
            if np.sum(mask) > 5:  # Verificar que hay suficientes puntos
                resist = self.resistencia_np[mask]
                v_final = self.v_finales_np[mask]
                
                # Agrupar por rangos de resistencia para reducir ruido
                bins = np.linspace(resist.min(), resist.max(), 30)
                indices = np.digitize(resist, bins)
                resist_binned = [resist[indices == i].mean() for i in range(1, len(bins)) if np.sum(indices == i) > 0]
                v_max_binned = [v_final[indices == i].max() for i in range(1, len(bins)) if np.sum(indices == i) > 0]
                
                # Ordenar por resistencia creciente
                if len(resist_binned) > 3:
                    sort_idx = np.argsort(resist_binned)
                    resist_binned = np.array(resist_binned)[sort_idx]
                    v_max_binned = np.array(v_max_binned)[sort_idx]
                    
                    # Suavizado con ventana móvil para evitar oscilaciones
                    window = min(5, len(resist_binned)//2)
                    if window > 1:
                        try:
                            v_smooth = savgol_filter(v_max_binned, window, 1)
                            ax.plot(resist_binned, v_smooth, color=color, linewidth=3, label=nombre)
                        except:
                            ax.plot(resist_binned, v_max_binned, color=color, linewidth=2, label=nombre)
                    else:
                        ax.plot(resist_binned, v_max_binned, color=color, linewidth=2, label=nombre)
        
        ax.set_xlabel("Resistencia [Ω]", fontsize=14)
        ax.set_ylabel("Velocidad final [m/s]", fontsize=14)
        ax.set_title("Velocidad vs Resistencia por tipo de amortiguamiento", fontsize=16)
        ax.grid(True)
        ax.legend(title="Tipo amortiguamiento", fontsize=12, loc='best')
        ax.set_xlim(0, 24)  # Limitar eje X de 0 a 24
        ax.set_xticks(np.linspace(0, 24, 25))  # 25 cotas: una por cada hora

    def crear_fig_superficie(self):
      """Gráfico 3: Superficie 3D y mapa 2D con interacción para mostrar l_bobina al hacer clic"""
      # Crear layout con dos paneles
      gs = self.fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
      ax3d = self.fig.add_subplot(gs[0], projection='3d')  # Panel izquierdo 3D
      ax2d = self.fig.add_subplot(gs[1])                  # Panel derecho 2D
      
      # Interpolación para todos los datos
      Xi, Yi, Vi = superficie_maxima(self.vueltas_np, self.d_hilo_np * 1000, self.v_finales_np, res=150)  # Convertir a mm
      _, _, Ri = superficie_maxima(self.vueltas_np, self.d_hilo_np * 1000, self.resistencia_np, res=150)  # Convertir a mm
      _, _, Li = superficie_maxima(self.vueltas_np, self.d_hilo_np * 1000, self.l_bobina_np, res=150)     # Convertir a mm
      
      # Suavizado
      Vi_smooth = gaussian_filter(Vi, sigma=1.5)
      Ri_smooth = gaussian_filter(Ri, sigma=1.5)
      Li_smooth = gaussian_filter(Li, sigma=1.5)
        
      # --- PANEL IZQUIERDO: SUPERFICIE 3D ---
      # Mapa de colores logarítmico
      norm = LogNorm(vmin=np.nanmax([np.nanmin(Ri_smooth), 1e-2]), vmax=np.nanmax(Ri_smooth))
      colors = cm.plasma(norm(Ri_smooth))
      
      # Reemplazar por rojo los valores de resistencia < 1 Ω
      rojo = np.array([1.0, 0.0, 0.0, 1.0])
      colors[Ri_smooth < 1.0] = rojo
      
      # Superficie 3D
      surf = ax3d.plot_surface(Xi, Yi, Vi_smooth, facecolors=colors,
                            rstride=1, cstride=1, linewidth=0, antialiased=True)
      
      # Cambiar los nombres de los ejes para mostrar mm en lugar de m
      ax3d.set_xlabel("Vueltas", fontsize=12)
      ax3d.set_ylabel("Diámetro de hilo (mm)", fontsize=12)  # Cambiado a mm
      ax3d.set_zlabel("Velocidad final (m/s)", fontsize=12)
      ax3d.set_title("Vista 3D", fontsize=14)
        
      # Leyenda de colores para 3D
      mappable = cm.ScalarMappable(norm=norm, cmap='plasma')
      mappable.set_array([])
      cbar = self.fig.colorbar(mappable, ax=ax3d, shrink=0.5, pad=0.1, label="Resistencia [Ω]")
      cbar.ax.text(0, -0.15, "Rojo: R < 1Ω", color='red', ha='left', fontsize=10)
      
      # --- PANEL DERECHO: VISTA 2D (MAPA DE CALOR) ---
      img = ax2d.pcolormesh(Xi, Yi, Vi_smooth, cmap='viridis', shading='auto')
      ax2d.set_xlabel("Vueltas", fontsize=12)
      ax2d.set_ylabel("Diámetro de hilo (mm)", fontsize=12)  # Cambiado a mm
      ax2d.set_title("Vista 2D (X-Y) - CLIC para ver información", fontsize=14)
      
      # Colorbar para el mapa 2D
      cbar2d = self.fig.colorbar(img, ax=ax2d, label="Velocidad final (m/s)")
      
      # --- INTERACTIVIDAD: CLIC MUESTRA INFORMACIÓN ---
      # Anotación para mostrar información
      annot = ax2d.annotate("", xy=(0,0), xytext=(20,20),
                          textcoords="offset points",
                          bbox=dict(boxstyle="round", fc="white", alpha=0.85),
                          arrowprops=dict(arrowstyle="->"))
      annot.set_visible(False)
      
      # Punto para marcar la ubicación seleccionada
      point, = ax2d.plot([], [], 'o', color='white', markersize=8, markeredgecolor='black', zorder=10)
      
      def update_info(x, y):
        """Actualiza la información en el punto (x,y)"""
        if x is not None and y is not None:
            # Obtener índices en la matriz interpolada
            i = min(max(0, int(np.argmin(abs(Xi[0, :] - x)))), Xi.shape[1]-1)
            j = min(max(0, int(np.argmin(abs(Yi[:, 0] - y)))), Yi.shape[0]-1)
            
            # Obtener valores interpolados
            vueltas = int(x)  # Número de vueltas
            d_hilo_mm = y      # Diámetro del hilo en mm
            d_hilo_m = d_hilo_mm / 1000  # Convertir a metros para cálculos
            velocidad = Vi_smooth[j, i]
            resistencia = Ri_smooth[j, i]
            l_bobina = Li_smooth[j, i]
            
            # Buscar el punto de datos real más cercano para valores exactos de otros parámetros
            idx_cercano = np.argmin((self.vueltas_np - vueltas)**2 + (self.d_hilo_np - d_hilo_m)**2)
            d_nucleo = self.d_nucleo_np[idx_cercano]
            l_nucleo = self.l_nucleo_np[idx_cercano]
            
            # --- CONSTANTES DEL SISTEMA (del código de simulación) ---
            area_transversal_mm2 = np.pi * (d_hilo_mm/2)**2
            
            # Energía inicial del sistema
            V0 = 400             # Tensión [V]
            C = 470e-6           # Capacitancia [F]
            m = 0.005            # Masa del proyectil [kg]
            energia_inicial = 0.5 * C * V0**2  # Julios
            
            # Energía cinética final y eficiencia
            energia_final = 0.5 * m * velocidad**2  # Julios
            eficiencia = (energia_final / energia_inicial) * 100  # Porcentaje
            
            mu_r = 1000          # Permeabilidad relativa
            densidad_nucleo = 7870  # [kg/m³]
            rho_cobre = 1.68e-8  # Resistividad del cobre [Ohm·m]
            
            # --- CÁLCULOS DE PARÁMETROS CONSTRUCTIVOS ---
            # Número de capas y bobinado
            espacio_hilo = d_hilo_m * 1.1  # 10% extra para espaciado entre hilos
            vueltas_por_capa = int(l_bobina / espacio_hilo)
            if vueltas_por_capa > 0:
                num_capas = int(np.ceil(vueltas / vueltas_por_capa))
            else:
                num_capas = "N/A"  # Evitar divisiones por cero
                
            # Diámetros de la bobina
            d_interior = d_nucleo  # Diámetro interior es el diámetro del núcleo
            if isinstance(num_capas, int):
                d_exterior = d_nucleo + (2 * num_capas * d_hilo_m)  # 2× por ambos lados
                d_bobina_medio = (d_interior + d_exterior) / 2
            else:
                d_exterior = "N/A"
                d_bobina_medio = "N/A"
                
            # Longitud del hilo y masa de la bobina
            if isinstance(num_capas, int) and isinstance(d_bobina_medio, float):
                # Calcular la longitud total del hilo sumando cada capa
                longitud_hilo_total = 0
                for capa in range(1, num_capas + 1):
                    d_capa = d_nucleo + (2 * (capa - 1) * d_hilo_m)
                    circunferencia = np.pi * d_capa
                    vueltas_en_capa = min(vueltas_por_capa, vueltas - ((capa - 1) * vueltas_por_capa))
                    if vueltas_en_capa > 0:  # Verificamos que no sea negativo
                        longitud_hilo_total += circunferencia * vueltas_en_capa
                        
                # Calcular volumen y masa del hilo (cobre)
                area_seccion = np.pi * (d_hilo_m/2)**2
                volumen_hilo = area_seccion * longitud_hilo_total
                densidad_cobre = 8960  # kg/m³
                masa_bobina = volumen_hilo * densidad_cobre * 1000  # en gramos
            else:
                longitud_hilo_total = "N/A"
                masa_bobina = "N/A"
                
# Energía almacenada en el capacitor
            energia_inicial = 0.5 * C * V0**2  # Julio
                
            # Posicionamiento responsivo con 9 cuadrantes
            x_min, x_max = ax2d.get_xlim()
            y_min, y_max = ax2d.get_ylim()

            # Dividir en 3 secciones cada eje
            x_tercio1 = x_min + (x_max - x_min) / 3
            x_tercio2 = x_min + 2 * (x_max - x_min) / 3
            y_tercio1 = y_min + (y_max - y_min) / 3
            y_tercio2 = y_min + 2 * (y_max - y_min) / 3
            
            # Matrices para posicionamiento
            matrizx = [100, -120, -200]  # [izquierda, centro, derecha]
            matrizy = [-50, -200, -330]  # [inferior, medio, superior]
            
            # Determinar índices de columna y fila
            if x < x_tercio1:  # Columna izquierda
                col_idx = 0
            elif x < x_tercio2:  # Columna media
                col_idx = 1
            else:  # Columna derecha
                col_idx = 2
                
            if y < y_tercio1:  # Fila inferior
                fila_idx = 0
            elif y < y_tercio2:  # Fila media
                fila_idx = 1
            else:  # Fila superior
                fila_idx = 2
            
            # Usar valores de las matrices para posicionar
            xytext = (matrizx[col_idx], matrizy[fila_idx])

            # Crear una NUEVA anotación con toda la información
            for texto in ax2d.texts:
                if hasattr(texto, '_es_info_tooltip') and texto._es_info_tooltip:
                    texto.remove()
            
            # Formatear números para mejor legibilidad
            def formato(valor, decimales=2, unidad=""):
                if isinstance(valor, (int, float)):
                    if isinstance(valor, int):
                        return f"{valor}{unidad}"
                    elif abs(valor) < 0.001:
                        return f"{valor*1000:.{decimales}f} m{unidad}"
                    else:
                        return f"{valor:.{decimales}f}{unidad}"
                else:
                    return str(valor)
                    
            # Formatear texto con toda la información constructiva
            info_texto = (
                f"--- DISEÑO DEL INDUCTOR ---\n"
                f"• Vueltas: {vueltas}\n"
                f"• Diám. hilo: {d_hilo_mm:.2f} mm\n"
                f"• Área transversal: {area_transversal_mm2:.2f} mm²\n"
                f"• Largo bobina: {l_bobina*100:.1f} cm\n"
                f"• Capas: {formato(num_capas)}\n"
                f"• Diám. interior: {d_interior*1000:.2f} mm\n"
                f"• Diám. exterior: {formato(d_exterior*1000 if isinstance(d_exterior, float) else d_exterior)} mm\n"
                f"• Diám. medio: {formato(d_bobina_medio*1000 if isinstance(d_bobina_medio, float) else d_bobina_medio)} mm\n"
                f"• Longitud hilo: {formato(longitud_hilo_total*100 if isinstance(longitud_hilo_total, float) else longitud_hilo_total)} cm\n"
                f"• Masa bobina: {formato(masa_bobina, unidad='g') if isinstance(masa_bobina, float) else masa_bobina}\n"
                f"\n--- PROYECTIL/NÚCLEO ---\n"
                f"• Diámetro: {d_nucleo*1000:.2f} mm\n"
                f"• Largo: {l_nucleo*100:.1f} cm\n" 
                f"• Masa: {m*1000:.1f} g\n"
                f"\n--- ELÉCTRICO/ENERGÉTICO ---\n"
                f"• Tensión: {V0} V\n"
                f"• Capacitancia: {C*1e6:.1f} µF\n"
                f"• Resistencia: {resistencia:.2f} Ω\n"
                f"• Energía inicial: {energia_inicial:.2f} J\n"
                f"• Energía final: {energia_final:.2f} J\n"
                f"• Eficiencia: {eficiencia:.1f}%\n"
                f"\n--- RESULTADO ---\n"
                f"• Velocidad: {velocidad:.2f} m/s"
            )
            
            annot = ax2d.annotate(info_texto,
                            xy=(x, y),
                            xytext=xytext,
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="white", alpha=0.9),
                            arrowprops=dict(arrowstyle="->"))
            
            # Marcar esta anotación para identificarla después
            annot._es_info_tooltip = True
            
            # Actualizar posición del punto marcador
            point.set_data([x], [y])
            
            # Redibujar
            self.fig.canvas.draw_idle()
            
      def on_click(event):
          """Manejador de eventos para clic"""
          if event.inaxes == ax2d:
              update_info(event.xdata, event.ydata)
      
      # Conectar evento de clic
      self.fig.canvas.mpl_connect("button_press_event", on_click)
      
      # Instrucciones para navegación
      self.fig.text(0.25, 0.95, 
                  "Arrastrar para rotar, Rueda para zoom, CLIC para ver datos",
                  ha='center', va='center', fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
      

    def crear_fig_nucleo(self):
        """Gráfico 4: Velocidad máxima vs d_nucleo"""
        ax = self.fig.add_subplot(111)
        
        dn_unique = np.unique(self.d_nucleo_np)
        max_v_por_dn = [self.v_finales_np[self.d_nucleo_np == dn].max() for dn in dn_unique]
        
        # Aplicar suavizado
        try:
            x_smooth, y_smooth = envolvente_suave(dn_unique, max_v_por_dn)
            ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label="Envolvente suavizada")
        except:
            pass  # Si falla el suavizado
            
        # Línea original
        ax.plot(dn_unique, max_v_por_dn, 'go-', alpha=0.7, label="Máximo v_final por d_nucleo")
        
        ax.set_xlabel("Diámetro núcleo (m)", fontsize=14)
        ax.set_ylabel("Velocidad final máxima (m/s)", fontsize=14)
        ax.set_title("Envolvente superior: Velocidad máxima vs d_nucleo", fontsize=16)
        ax.grid(True)
        ax.legend(fontsize=12)

if __name__ == "__main__":
    visualizador = VisualizadorGraficos()
    plt.show()