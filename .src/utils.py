import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, make_interp_spline


def superficie_maxima_cacheada(x, y, z, res=100, cache_file="simulation_caché/superficie_cache.npz"):
    """Crea o carga una superficie interpolada de los máximos"""
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        Xi, Yi, Zi = data["Xi"], data["Yi"], data["Zi"]
    else:
        Xi, Yi, Zi = superficie_maxima(x, y, z, res)
        np.savez(cache_file, Xi=Xi, Yi=Yi, Zi=Zi)
    return Xi, Yi, Zi

def cargar_datos():
    """Carga los datos desde CSV"""
    print("Cargando datos de simulación...")
    df = pd.read_csv("data/resultados_simulacion.csv")
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