# 📚 Diseño de Clases para Simulador Gauss

## 1. Clase: InductanciaBase
**Responsabilidad:** Calcular parámetros constructivos de la bobina y la inductancia \( L_0 \).

- **Inputs (constructor):**
  - espacio_disponible_largo: longitud total disponible [m]
  - espacio_disponible_diametro: diámetro total disponible [m]
  - d_hilo: diámetro del hilo conductor [m]
  - N_deseado: cantidad de vueltas objetivo (opcional, si no se calcula automáticamente)
  - mu_r: permeabilidad relativa del núcleo (default 1, aire)

- **Propiedades Calculadas (atributos):**
  - vueltas_por_capa: cuántas vueltas entran en una capa
  - cantidad_de_capas: número de capas necesarias
  - largo_bobina (l_b): largo real de la bobina [m]
  - diametro_medio: diámetro medio de la bobina [m]
  - area_transversal (A): área transversal de la bobina [m²]
  - longitud_total_alambre (l_c): longitud total del cable usado [m]
  - area_alambre (A_c): área de sección transversal del hilo de cobre [m²]

- **Métodos:**
  - `calcular_parametros_constructivos()`
    - Organiza vueltas en capas y largo.
    - Calcula diámetro medio: \( d_{\text{medio}} = \text{promedio entre diámetro interno y externo} \)
    - Calcula área de sección \( A \) (depende de \( d_{\text{medio}} \)).
    - Calcula longitud total de alambre \( l_c \).

  - `calcular_L0() -> float`
    - Fórmula:
      \[
      L_0 = \mu_0 \mu_r \frac{N^2 A}{l_b}
      \]

---

## 2. Clase: InductanciaDinamica
**Responsabilidad:** Calcular \( L(x(t)) \) en función de la posición del núcleo.

- **Inputs:**
  - instancia de `InductanciaBase`
  - x(t): posición del núcleo móvil

- **Métodos:**
  - `calcular_L(x: float) -> float`
    - Fórmula: 
      \[
      L(x) = \frac{N^2 \mu_0 A}{l_b - x + \frac{x}{\mu_r}}
      \]

---

## 3. Clase: CampoMagnetico
**Responsabilidad:** Calcular \( B(t) \) (campo magnético).

- **Inputs:**
  - instancia de `InductanciaDinamica` o `InductanciaBase`
  - corriente I(t)

- **Métodos:**
  - `calcular_B(I: float, x: float) -> float`
    - Fórmula: 
      \[
      B(t) = \mu_0 \mu_r \frac{N}{l_b} I(t)
      \]

---

## 4. Clase: Resistencia
**Responsabilidad:** Calcular la resistencia R del alambre.

- **Inputs:**
  - rho_c: resistividad del cobre
  - l_c: longitud del cable
  - A_c: área de sección transversal del conductor

- **Métodos:**
  - `calcular_R() -> float`
    - Fórmula:
      \[
      R = \rho_c \frac{l_c}{A_c}
      \]

---

## 5. Clase: CorrienteCircuito
**Responsabilidad:** Calcular \( I(t) \) en circuito RLC (y decidir régimen).

- **Inputs:**
  - V_0: tensión inicial
  - instancia de `InductanciaDinamica`
  - instancia de `Resistencia`
  - C: capacitancia

- **Métodos:**
  - `calcular_I(t: float) -> float`
    - Resolver:
      - Subamortiguado: 
        \[
        I(t) = I_0 e^{-\alpha t} \sin(\omega_d t)
        \]
      - Críticamente amortiguado:
        \[
        I(t) = (A + Bt)e^{-\alpha t}
        \]
      - Sobreamortiguado:
        \[
        I(t) = A e^{s_1 t} + B e^{s_2 t}
        \]
    - Cálculo de constantes:
      - 
        \[
        \omega_0 = \frac{1}{\sqrt{L C}}
        \]
      - 
        \[
        \alpha = \frac{R}{2L}
        \]
      - Evaluación de:
        \[
        \Delta = \left( \frac{R}{2L} \right)^2 - \frac{1}{LC}
        \]
        para decidir el régimen.

---

# 📈 Relaciones entre Clases

```plaintext
InductanciaBase --> usada por --> InductanciaDinamica
InductanciaDinamica + CorrienteCircuito --> usadas por --> CampoMagnetico
Resistencia --> usada por --> CorrienteCircuito
```

[InductanciaBase] 
  --> calcular_parametros_constructivos()
    - vueltas_por_capa
    - cantidad_de_capas
    - largo_bobina (l_b)
    - diametro_medio
    - area_transversal (A)
    - longitud_total_alambre (l_c)
    - area_alambre (A_c)

  --> calcular_L0()
    - L0 = (mu0 * mu_r * N^2 * A) / l_b

[Resistencia]
  --> calcular_R()
    - R = (rho_cobre * l_c) / A_c

[PerdidasNucleo]
  --> calcular_ke()
    - ke ~ d_n^2 / rho_nucleo

[CorrienteCircuito]
  --> calcular constantes:
    - omega0 = 1 / sqrt(L0 * C)
    - alpha = R / (2 * L0)
    - delta = alpha^2 - omega0^2

    - Si delta < 0 (subamortiguado):
        omega_d = sqrt(omega0^2 - alpha^2)

🕓 Etapa 1: En cada paso de simulación (para cada instante t)

[CorrienteCircuito]
  --> calcular_I(t)
    - Si subamortiguado:
      I(t) = I0 * e^(-alpha * t) * sin(omega_d * t)
    - Si críticamente amortiguado:
      I(t) = (A + B*t) * e^(-alpha * t)
    - Si sobreamortiguado:
      I(t) = A * e^(s1 * t) + B * e^(s2 * t)

[CampoMagnetico]
  --> calcular_B(I(t), x(t))
    - B(t) = (mu0 * mu_r * N / l_b) * I(t)

[InductanciaDinamica]
  --> calcular_L(x(t))
    - L(x) = (N^2 * mu0 * A) / (l_b - x + (x / mu_r))

Opcional:
  --> calcular energía magnética:
    - W_mag(t) = (B(t)^2 / (2 * mu0 * mu_r)) * volumen
