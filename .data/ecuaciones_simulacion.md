# Descripción matemática del código de simulación de cañón Gauss

Este documento describe, línea a línea, las ecuaciones y relaciones físicas implementadas en el archivo `math.py`. Cada bloque corresponde a la sección de código paralela, explicando en notación matemática y física el significado de cada instrucción.

---

## 1. Definición de constantes físicas y parámetros de entrada

- **Capacitancia**
  
  $C = 470 \times 10^{-6}\ \mathrm{F}$

- **Tensión inicial**
  
  $V_0 = 400\ \mathrm{V}$

- **Corriente inicial**
  
  $I_0 = 0\ \mathrm{A}$

- **Resistividad del cobre**
  
  $\rho_{Cu} = 1.7 \times 10^{-8}\ \Omega\cdot\mathrm{m}$

- **Resistividad del núcleo**
  
  $\rho_{n} = 0.5 \times 10^{-6}\ \Omega\cdot\mathrm{m}$

- **Área del lazo de histéresis**
  
  $ABH = 250\ \mathrm{J/m^3}$

- **Inducción de saturación**
  
  $B_s = 1.5\ \mathrm{T}$

- **Densidad del cobre**
  
  $\sigma_{Cu} = 8960\ \mathrm{kg/m^3}$

- **Densidad del núcleo**
  
  $\sigma_{n} = 7650\ \mathrm{kg/m^3}$

- **Masa del núcleo**
  
  $m_n = 0.003\ \mathrm{kg}$

- **Permeabilidad relativa del núcleo**
  
  $\mu_{r,n} = 5000$

- **Permeabilidad del vacío**
  
  $\mu_0 = 4\pi \times 10^{-7}\ \mathrm{H/m}$

- **Paso temporal**
  
  $\Delta t = 1 \times 10^{-4}\ \mathrm{s}$

---

## 2. Parámetros geométricos de la bobina y núcleo

- **Número de vueltas**
  
  $N = 100$

- **Diámetro del hilo de cobre**
  
  $d_{Cu} = 0.001\ \mathrm{m}$

- **Diámetro del núcleo**
  
  $d_n = 0.005\ \mathrm{m}$

- **Longitud de la bobina**
  
  $l_b = 0.1\ \mathrm{m}$

---

## 3. Constantes fijas calculadas

- **Volumen del núcleo**
  
  $V_n = \frac{m_n}{\sigma_n}$

- **Área de la sección del cobre**
  
  $A_{Cu} = \pi \left(\frac{d_{Cu}}{2}\right)^2$

- **Área de la sección del núcleo**
  
  $A_n = \pi \left(\frac{d_n}{2}\right)^2$

- **Número de capas**
  
  $n_{capas} = -\left(\left\lceil -\frac{N \cdot d_{Cu}}{l_b} \right\rceil\right)$

- **Vueltas por capa**
  
  $N_{capa} = \left\lfloor \frac{l_b}{d_{Cu}} \right\rfloor$

- **Vueltas en la capa final**
  
  $N_{final} = N - (n_{capas} - 1) N_{capa}$

- **Longitud total de cobre**
  
  $L_{Cu} = \sum_{i=0}^{n_{capas}-1} N_{i} \cdot \pi \cdot d_{i}$
  
  donde $d_{i} = d_n + 2 d_{Cu} i$ y $N_{i}$ es el número de vueltas en la capa $i$ ($N_{capa}$ para capas completas, $N_{final}$ para la última).

- **Diámetro medio de la bobina**
  
  $d_{med} = \text{promedio ponderado de los diámetros de cada capa}$

- **Constante de Foucault**
  
  $K_E = \frac{d_n^2}{\rho_n}$

- **Resistencia del cobre**
  
  $R_{Cu} = \frac{\rho_{Cu} L_{Cu}}{A_{Cu}}$

- **Inductancia inicial**
  
  $L_0 = \frac{\mu_0 \mu_{r,n} N^2 A_{med}}{l_b}$

- **Centro de la bobina**
  
  $x_{centro} = \frac{l_b}{2}$

---

## 4. Ecuaciones diferenciales del circuito RLC

- **Constante de amortiguamiento**
  
  $\alpha = \frac{R_{Cu}}{2 L_0}$

- **Frecuencia angular natural**
  
  $\omega_0 = \frac{1}{\sqrt{L_0 C}}$

- **Discriminante**
  
  $\Delta = \alpha^2 - \omega_0^2$

---

## 5. Soluciones para la corriente y tensión

- **Subamortiguado:**
  
  $I(t) = I_0 e^{-\alpha t} \sin(\omega_d t)$
  
  $V(t) = V_0 e^{-\alpha t} \left[\cos(\omega_d t + \phi)\right]^2/2$

- **Críticamente amortiguado:**
  
  $I(t) = (A + B t) e^{-\alpha t}$

- **Sobreamortiguado:**
  
  $I(t) = A e^{s_1 t} + B e^{s_2 t}$

---

## 6. Cálculo de variables dinámicas en el bucle de simulación

- **Energía magnética máxima disponible (limitada por saturación)**
  
  $W_{mag,\,max} = \min\left(W_{el} - W_{Joule} - W_{Foucault} - W_{hyst} - W_{kin},\ w_{sat}\right)$

- **Fuerza magnética**
  
  $\Delta x = x_{n} - x_{n-1}$
  
  $F_{mag} = \begin{cases}
    0 & \text{si } |\Delta x| < \varepsilon \\
    \frac{W_{mag,\,max} - W_{mag,\,ant}}{\Delta x} & \text{si } |\Delta x| \geq \varepsilon
  \end{cases}$

- **Aceleración**
  
  $a = \frac{F_{mag}}{m_n}$

- **Velocidad y posición**
  
  $v_{n+1} = v_n + a \Delta t$
  
  $x_{n+1} = x_n + v_{n+1} \Delta t$

- **Actualización de energía magnética anterior**
  
  $W_{mag,\,ant} \leftarrow W_{mag,\,max}$

---

## 7. Estadísticas y balance de energía

- **Corriente pico**
  
  $I_{max} = \max |I(t)|$

- **Balance de energía**
  
  Se imprime el desglose de cada término energético y su porcentaje respecto a la energía inicial del condensador.

---

Este documento puede usarse como referencia paralela al código fuente para entender la correspondencia entre cada línea de código y su fundamento físico-matemático.
