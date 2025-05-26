import os
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

import cupy as cp
import numpy as np
import time
import pandas as pd

csv_path = ".data/resultados_simulacion.csv"

def simulador():
    # --- Constantes físicas ---
    mu0 = 4 * np.pi * 1e-7
    mu_r = 1000
    rho_cobre = 1.68e-8
    C = 470e-6
    V0 = 400
    densidad_nucleo = 7870  # kg/m³ (ajusta según material)
    m = 0.005     # kg (ajusta según tu caso)
    dt = 5*1e-5 #este y el de abajo varían cuanto error hay :/
    n_steps = 100

    # --- Parámetros de diseño ---
    vueltas = cp.arange(10, 801, 1, dtype=cp.float32)           #aumentale el paso a 30-50 para pruebas, 5 para usarse, 1 si ya es cálculo final
    d_hilo = cp.arange(0.0001, 0.0015, 0.0001, dtype=cp.float32)  
    d_nucleo = cp.arange(0.004, 0.01, 0.001, dtype=cp.float32)     
    l_bobina = cp.arange(0.01, 0.4, 0.01, dtype=cp.float32)      

    # --- Malla de combinaciones ---
    N, DH, DN, LB = cp.meshgrid(vueltas, d_hilo, d_nucleo, l_bobina, indexing='ij')
    total_comb = N.size
    print(f"Simulando {total_comb:,} combinaciones...")

    # --- Aplanar ---
    N = N.ravel()
    DH = DH.ravel()
    DN = DN.ravel()
    LB = LB.ravel()

    LN = m / (densidad_nucleo * cp.pi * (DN / 2)**2)

    # --- Resultados ---
    v_finales = []
    parametros = []

    # --- Bloques más grandes (VRAM) ---
    block_size = 100000
    start_time = time.time()
    t_cpu = np.arange(0, n_steps * dt, dt, dtype=np.float32)
    t_gpu = cp.asarray(t_cpu)


#! REVISÁ A PARTIR DE ACÁ, LO HIZO TODO GPT

#* Se calcula el inductor
#* Se calcula la resistencia
#* Se calcula la inductancia
#* Se selecciona el tipo de amortiguamiento
#* En función de esto, se usa la respectiva ecuación
#* Se calculan los parámetros dinámicos 
#* Campo, inductancia, corriente y tensión
#* Fuerza, aceleración, velocidad y posición

    # --- Procesamiento por bloques ---
    for i in range(0, total_comb, block_size):
        end = min(i + block_size, total_comb)
        blk_len = end - i

        # Extraer bloque
        n = N[i:end]
        d_h = DH[i:end]
        d_n = DN[i:end]
        l_n = LN[i:end]
        l_b = LB[i:end]
    # --- Cálculo realista de bobinado ---
        # Vueltas por capa (aprox)
        vueltas_por_capa = cp.floor(l_b / d_h)
        # Número de capas necesarias
        capas = cp.ceil(n / vueltas_por_capa)
        # Diámetro exterior de la última capa
        d_ext = d_n + 2 * d_h * capas
        # Diámetro medio real para inductancia y resistencia
        d_medio = (d_n + d_ext) / 2

        # Parámetros derivados usando d_medio
        A_cobre = cp.pi * (d_h / 2)**2
        l_cobre = n * cp.pi * d_medio
        R = (rho_cobre * l_cobre) / A_cobre
        A_medio = cp.pi * (d_medio / 2)**2
        L0 = mu0 * mu_r * n**2 * A_medio / l_b

        # Inicialización
        I = cp.zeros((blk_len, n_steps), dtype=cp.float32)
        B = cp.zeros((blk_len, n_steps), dtype=cp.float32)
        L = cp.zeros((blk_len, n_steps), dtype=cp.float32)
        a = cp.zeros((blk_len, n_steps), dtype=cp.float32)
        v = cp.zeros((blk_len, n_steps), dtype=cp.float32)
        x = cp.zeros((blk_len, n_steps), dtype=cp.float32)
        
        # Guardar tipo de amortiguamiento
        tipo_amortiguamiento = cp.zeros(blk_len, dtype=cp.int8)

        B[:, 0] = 1e-9
        L[:, 0] = L0

        # Loop temporal vectorizado
        for k in range(1, n_steps):
            # Actualizar inductancia con posición actual
            L[:, k] = mu0 * n**2 * A_medio / (l_b - x[:, k-1] + x[:, k-1] / mu_r)
            alpha = R / (2 * L[:, k])
            omega_0 = 1 / cp.sqrt(L[:, k] * C)
            delta = alpha**2 - omega_0**2

            # Considerar crítico si delta está diferencialmente cerca de cero
            d0 = 1  # Umbral para considerar crítico
            if k == 1:

                tipo_amortiguamiento = cp.where(cp.abs(delta) < d0, 1,    # crítico
                                    cp.where(delta < 0, 0, 2))        # subamortiguado, sobreamortiguado
            omega_d = cp.sqrt(cp.maximum(omega_0**2 - alpha**2, 1e-12))

            # --- Tensión variable solo para subamortiguado ---
            # Vc(t) = V0 * exp(-alpha*t) * [cos(omega_d*t) + (alpha/omega_d) * sin(omega_d*t)]
            Vc = V0 * cp.exp(-alpha * t_gpu[k]) * (
                cp.cos(omega_d * t_gpu[k]) + (alpha / omega_d) * cp.sin(omega_d * t_gpu[k])
            )

            # Subamortiguado: I(t) = (V0/(L*omega_d)) * exp(-alpha*t) * sin(omega_d*t)
            # Usar Vc(t) en vez de V0
            I_sub = Vc / (L[:, k] * omega_d) * cp.exp(-alpha * t_gpu[k]) * cp.sin(omega_d * t_gpu[k])

            # Críticamente amortiguado: I(t) = (V0/L) * t * exp(-alpha*t)
            I_crit = V0 / L[:, k] * t_gpu[k] * cp.exp(-alpha * t_gpu[k])

            # Sobreamortiguado: I(t) = A*exp(s1*t) + B*exp(s2*t)
            s1 = -alpha + cp.sqrt(cp.maximum(alpha**2 - omega_0**2, 1e-12))
            s2 = -alpha - cp.sqrt(cp.maximum(alpha**2 - omega_0**2, 1e-12))
            denom = L[:, k] * (s1 - s2)
            denom = cp.where(cp.abs(denom) < 1e-12, 1e-12, denom)
            A_coef = -V0 / denom
            B_coef = V0 / denom
            I_sobre = A_coef * cp.exp(s1 * t_gpu[k]) + B_coef * cp.exp(s2 * t_gpu[k])

            # --- Selección de ecuación según tipo ---
            I[:, k] = cp.where(tipo_amortiguamiento == 0, I_sub, I[:, k])
            I[:, k] = cp.where(tipo_amortiguamiento == 1, I_crit, I[:, k])
            I[:, k] = cp.where(tipo_amortiguamiento == 2, I_sobre, I[:, k])

            B[:, k] = mu0 * mu_r * n * I[:, k] / l_b
            F = B[:, k] * I[:, k] * A_medio
            a[:, k] = F / m
            v[:, k] = v[:, k-1] + a[:, k] * dt
            x[:, k] = x[:, k-1] + v[:, k] * dt

#!HASTA ACÁ
        # Guardar resultados
        v_block = v[:, -1].get()
        r_block = R.get()
        tipo_amort_block = tipo_amortiguamiento.get()
        
        v_finales.extend(v_block.tolist())
        for j in range(blk_len):
            parametros.append((
                int(n[j].get()), float(d_h[j].get()), float(d_n[j].get()),
                float(l_n[j].get()), float(l_b[j].get()), float(r_block[j]),
                int(tipo_amort_block[j])
            ))

        # Progreso
        print(f"Simuladas {end}/{total_comb} combinaciones ({100*end/total_comb:.1f}%) - {time.time() - start_time:.2f}s")

    print(f"\nTiempo total de ejecución: {time.time() - start_time:.2f} s")

    # Convertir resultados a arrays de numpy
    v_finales_np = np.array(v_finales)
    parametros_np = np.array(parametros)

    # --- FILTRAR NaN ---
    validos = ~np.isnan(v_finales_np)
    v_finales_np = v_finales_np[validos]
    parametros_np = parametros_np[validos]

    # Extraer columnas
    vueltas_np = parametros_np[:, 0]
    d_hilo_np = parametros_np[:, 1]
    d_nucleo_np = parametros_np[:, 2]
    l_nucleo_np = parametros_np[:, 3]
    l_bobina_np = parametros_np[:, 4]
    resistencia_np = parametros_np[:, 5]
    tipo_amort_np = parametros_np[:, 6]

        # Crear DataFrame con los resultados
    df = pd.DataFrame({
        "v_final": v_finales_np,
        "N": vueltas_np,
        "d_hilo": d_hilo_np,
        "d_nucleo": d_nucleo_np,
        "l_nucleo": l_nucleo_np,
        "l_bobina": l_bobina_np,
        "R": resistencia_np,
        "tipo_amort": tipo_amort_np.astype(int)
    })

    # Guardar a CSV
    df.to_csv(".data/resultados_simulacion.csv", index=False)
    print("Resultados guardados en resultados_simulacion.csv")

    return v_finales_np, vueltas_np, d_hilo_np, d_nucleo_np, l_nucleo_np, l_bobina_np, resistencia_np, tipo_amort_np

def graph(v_finales_np, vueltas_np, d_hilo_np, d_nucleo_np, l_nucleo_np, l_bobina_np, resistencia_np, tipo_amort_np):

    # --- Top 10 usando arrays de numpy (coherente con el gráfico) ---
    indices_top10 = np.argsort(v_finales_np)[::-1][:10]
    print("Cantidad de NaN en v_finales_np:", np.isnan(v_finales_np).sum())
    print("Máximo v_finales_np:", np.nanmax(v_finales_np))
    print("Mínimo v_finales_np:", np.nanmin(v_finales_np))
    print("\nTop 10 combinaciones por velocidad final:")
    print("v_final [m/s] |  N  | d_hilo  | d_nucleo | l_nucleo | l_bobina |  R [Ω]  | Amortiguamiento")
    print("--------------------------------------------------------------------------------------")
    for idx in indices_top10:
        v = v_finales_np[idx]
        n = int(vueltas_np[idx])
        dh = d_hilo_np[idx]
        dn = d_nucleo_np[idx]
        ln = l_nucleo_np[idx]
        lb = l_bobina_np[idx]
        r = resistencia_np[idx]
        tipo = int(tipo_amort_np[idx])
        tipo_str = "Subamort." if tipo == 0 else "Crítico" if tipo == 1 else "Sobreamort."
        print(f"{v:.6f}     | {n:3d} | {dh:.5f} | {dn:.5f}   | {ln:.5f}   | {lb:.5f} | {r:.4f} | {tipo_str}")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata, make_interp_spline
    from scipy.ndimage import gaussian_filter

    def superficie_maxima(x, y, z, res=100):
        # Para cada (x, y) único, tomar el máximo z
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
        x = np.array(x)
        y = np.array(y)
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        x_smooth = np.linspace(x.min(), x.max(), puntos)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth
    fig = plt.figure(figsize=(16, 10))

    # --- 1. Línea envolvente: Velocidad máxima vs Vueltas ---
    ax1 = fig.add_subplot(2, 2, 1)
    vueltas_unique = np.unique(vueltas_np)
    max_v_por_vuelta = [v_finales_np[vueltas_np == v].max() for v in vueltas_unique]
    ax1.plot(vueltas_unique, max_v_por_vuelta, label="Máximo v_final por vueltas")
    ax1.set_xlabel("Número de vueltas")
    ax1.set_ylabel("Velocidad final máxima (m/s)")
    ax1.set_title("Envolvente superior: Velocidad máxima vs Vueltas")
    ax1.grid(True)
    ax1.legend()

    # --- 2. Gráfico de amortiguamiento suavizado ---
    ax2 = fig.add_subplot(2, 2, 2)
    categorias = ['Subamortiguado', 'Crítico', 'Sobreamortiguado']
    colores = ['blue', 'green', 'red']

    # Procesar cada tipo de amortiguamiento por separado
    for tipo_amort, color, nombre in zip([0, 1, 2], colores, categorias):
        # Filtrar solo este tipo
        mask = tipo_amort_np == tipo_amort
        if np.sum(mask) > 5:  # Verificar que hay suficientes puntos
            resist = resistencia_np[mask]
            v_final = v_finales_np[mask]
            
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
                    from scipy.signal import savgol_filter
                    try:
                        v_smooth = savgol_filter(v_max_binned, window, 1)
                        ax2.plot(resist_binned, v_smooth, color=color, linewidth=2, label=nombre)
                    except:
                        # Si falla el filtro, unir puntos directamente
                        ax2.plot(resist_binned, v_max_binned, color=color, linewidth=2, label=nombre)
                else:
                    ax2.plot(resist_binned, v_max_binned, color=color, linewidth=2, label=nombre)
                

    ax2.set_xlabel("Resistencia [Ω]")
    ax2.set_ylabel("Velocidad final [m/s]")
    ax2.set_title("Velocidad vs Resistencia por tipo de amortiguamiento")
    ax2.grid(True)
    ax2.legend(title="Tipo amortiguamiento")


    # --- 3. Superficie continua: v_final vs Vueltas y d_hilo (más suave) ---

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')

    # Usar la función superficie_maxima para obtener solo los máximos (elimina ruido)
    Xi, Yi, Vi = superficie_maxima(vueltas_np, d_hilo_np, v_finales_np, res=150)

    # Aplicar un filtro gaussiano para suavizar aún más
    Vi_smooth = gaussian_filter(Vi, sigma=1.5)

    # Graficar con más suavidad en la representación
    surf = ax3.plot_surface(Xi, Yi, Vi_smooth, cmap='plasma', 
                        alpha=1, antialiased=True,
                        rstride=1, cstride=1, 
                        linewidth=0)

    ax3.set_xlabel("Vueltas")
    ax3.set_ylabel("Diámetro de hilo (m)")
    ax3.set_zlabel("Velocidad final (m/s)")
    ax3.set_title("Superficie interpolada suavizada: v_final vs Vueltas y d_hilo")
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=10, label="Velocidad final [m/s]")
    # --- 4. Línea envolvente: Velocidad máxima vs d_nucleo ---
    ax4 = fig.add_subplot(2, 2, 4)
    dn_unique = np.unique(d_nucleo_np)
    max_v_por_dn = [v_finales_np[d_nucleo_np == dn].max() for dn in dn_unique]
    ax4.plot(dn_unique, max_v_por_dn, color='green', label="Máximo v_final por d_nucleo")
    ax4.set_xlabel("Diámetro núcleo (m)")
    ax4.set_ylabel("Velocidad final máxima (m/s)")
    ax4.set_title("Envolvente superior: Velocidad máxima vs d_nucleo")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()

if os.path.exists(csv_path):
    print("Cargando datos de simulación desde CSV...")
    df = pd.read_csv(csv_path)
    v_finales_np = df["v_final"].values
    vueltas_np = df["N"].values
    d_hilo_np = df["d_hilo"].values
    d_nucleo_np = df["d_nucleo"].values
    l_nucleo_np = df["l_nucleo"].values
    l_bobina_np = df["l_bobina"].values
    resistencia_np = df["R"].values
    tipo_amort_np = df["tipo_amort"].values
else: v_finales_np, vueltas_np, d_hilo_np, d_nucleo_np, l_nucleo_np, l_bobina_np, resistencia_np, tipo_amort_np = simulador()
pass
graph(v_finales_np, vueltas_np, d_hilo_np, d_nucleo_np, l_nucleo_np, l_bobina_np, resistencia_np, tipo_amort_np)