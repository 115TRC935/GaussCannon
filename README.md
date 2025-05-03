# ⚡ GaussCannon - Simulación y Visualización

Bienvenido al proyecto **GaussCannon**: una herramienta para simular y analizar el comportamiento de cañones de Gauss, con visualizaciones interactivas y análisis de parámetros constructivos.

---

## 📋 Índice

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Uso rápido](#uso-rápido)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Visualizaciones](#visualizaciones)
- [Parámetros simulados](#parámetros-simulados)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

---

## 📖 Descripción

**GaussCannon** permite simular el comportamiento físico de un cañón de Gauss, generando datos y visualizaciones para analizar el impacto de distintos parámetros constructivos. Incluye herramientas para visualizar resultados en gráficos 2D y superficies 3D interactivas.

---

## 🚀 Instalación

1. Cloná este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/GaussCannon.git
   cd GaussCannon
   ```
2. Instalá las dependencias de Python:
   ```bash
   pip install -r requirements.txt
   ```
   > **Nota:** Se requiere Python 3.7+ y las librerías listadas en `requirements.txt` (incluye `numpy`, `plotly`, etc).

---

## ⚡ Uso rápido

### Simulación y visualización de resultados

1. **Generar datos de simulación**  
   Ejecutá el script principal para simular y guardar los resultados:
   ```bash
   python classes.py
   ```
   Esto generará el archivo `resultados_simulacion.csv`.

2. **Visualizar resultados en gráficos interactivos**  
   Ejecutá el visualizador:
   ```bash
   python grafico_completo.py
   ```
   > Navegá entre los distintos gráficos usando las flechas del teclado o los botones en pantalla.

3. **Visualización 3D pre-renderizada**  
   Para generar y abrir la superficie 3D interactiva:
   ```bash
   python .src/ploterWebPreRenderizado.py
   ```
   Esto creará el archivo `.src/superficie3d_interactiva.html` que podés abrir en tu navegador para explorar la superficie 3D de velocidad final en función de vueltas y diámetro de hilo.

---

## 🗂️ Estructura del proyecto

```
GaussCannon/
│
├── classes.py                   # Simulación física y generación de datos
├── grafico_completo.py          # Visualización interactiva de resultados 2D/3D
├── .src/
│   ├── ploterWebPreRenderizado.py   # Genera el HTML interactivo de la superficie 3D
│   └── superficie3d_interactiva.html # Visualización 3D pre-renderizada (autogenerado)
│   └── simulation_caché/            # Caché de superficies simuladas (usado por el ploter)
├── resultados_simulacion.csv    # Resultados de simulación (autogenerado)
├── requirements.txt
├── ecuaciones.pdf
└── README.md
```

---

## 📊 Visualizaciones

- **Velocidad máxima vs Vueltas**
- **Velocidad vs Resistencia por tipo de amortiguamiento**
- **Superficie 3D: Velocidad vs Vueltas y diámetro de hilo**  
  (Interactiva, generada en `.src/superficie3d_interactiva.html`)
- **Velocidad máxima vs diámetro del núcleo**

Navega entre gráficos con las flechas del teclado o los botones en pantalla.

---

## ⚙️ Parámetros simulados

- Número de vueltas
- Diámetro del hilo y núcleo
- Longitud de bobina y núcleo
- Resistencia eléctrica
- Tipo de amortiguamiento (subamortiguado, crítico, sobreamortiguado)
- Energía y eficiencia

---

## 🤝 Contribuir

¡Pull requests y sugerencias son bienvenidas!  
Por favor, abrí un issue para discutir cambios importantes.

---

## 📄 Licencia

MIT License © 2024

---
