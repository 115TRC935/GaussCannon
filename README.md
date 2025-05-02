# ⚡ GaussCannon - Simulación y Visualización

Bienvenido al proyecto **GaussCannon**: una herramienta para simular y analizar el comportamiento de cañones de Gauss, con visualizaciones interactivas y análisis de parámetros constructivos.

---

## 📋 Índice

1. [Descripción](#descripción)
2. [Instalación](#instalación)
3. [Uso rápido](#uso-rápido)
4. [Estructura del proyecto](#estructura-del-proyecto)
5. [Visualizaciones](#visualizaciones)
6. [Parámetros simulados](#parámetros-simulados)
7. [Contribuir](#contribuir)
8. [Licencia](#licencia)

---

## 📖 Descripción

Este proyecto permite:
- Simular el rendimiento de un cañón de Gauss variando parámetros físicos y constructivos.
- Analizar resultados y visualizar datos en gráficos interactivos.
- Obtener recomendaciones de diseño óptimo.

---

## 🚀 Instalación

1. Cloná este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/GaussCannon.git
   cd GaussCannon
   ```
2. Instalá las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ Uso rápido

Ejecutá el visualizador de gráficos:
```bash
python grafico_completo.py
```
> Asegurate de tener el archivo `resultados_simulacion.csv` generado o ejecutá primero `classes.py` para simular.

---

## 🗂️ Estructura del proyecto

```
GaussCannon/
│
├── classes.py                # Simulación física y generación de datos
├── grafico_completo.py       # Visualización interactiva de resultados
├── resultados_simulacion.csv # Resultados de simulación (autogenerado)
├── requirements.txt
├── ecuaciones.pdf
└── README.md
```

---

## 📊 Visualizaciones

- **Velocidad máxima vs Vueltas**
- **Velocidad vs Resistencia por tipo de amortiguamiento**
- **Superficie 3D: Velocidad vs Vueltas y diámetro de hilo**
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
