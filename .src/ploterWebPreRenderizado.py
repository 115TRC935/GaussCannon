import numpy as np
import plotly.graph_objects as go

# Cargar la superficie cacheada
data = np.load("simulation_caché/superficie_vf_cache.npz")
Xi, Yi, Zi = data["Xi"], data["Yi"], data["Zi"]

# Crear la figura de superficie
fig = go.Figure(data=[go.Surface(z=Zi, x=Xi, y=Yi, colorscale='Plasma')])

fig.update_layout(
    title="Vista 3D",
    scene=dict(
        xaxis_title="Vueltas",
        yaxis_title="Diámetro de hilo (mm)",
        zaxis_title="Velocidad final (m/s)",
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            tickfont=dict(size=12)
        ),
        zaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            tickfont=dict(size=12)
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        ),
        aspectmode='manual',
        aspectratio=dict(x=1.2, y=1, z=0.7)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    font=dict(family="Arial", size=14),
)

# Guardar como imagen y HTML interactivo
fig.write_html(".src/superficie3d_interactiva.html")

# Mostrar en el navegador
fig.show()