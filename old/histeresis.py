import math

class HisteresisHierroSilicio:
    def __init__(self):
        self.mu_r_max = 4000  # Permeabilidad relativa máxima

    def B_saturacion(self, masa):
        """
        Ecuación: B_sat = 1.5 * (masa / 1e-3) ** 0.3
        - Relaciona la saturación magnética del material con su masa.
        - A menor masa, menor saturación magnética.
        """
        masa = max(masa, 1e-6)  # Evitar log(0)
        return 1.5 * (masa / 1e-3) ** 0.3  # Saturación baja si la masa es baja

    def mu_r(self, B, masa):
        """
        Ecuación: mu_r = mu_r_max * tanh(B / B_sat)
        - Calcula la permeabilidad relativa del material.
        - Depende del campo magnético B y de la saturación magnética B_sat.
        - Usa la función tanh para modelar la no linealidad de la histéresis.
        """
        B_sat = self.B_saturacion(masa)
        return self.mu_r_max * math.tanh(B / B_sat)

    def B(self, H, masa):
        """
        Ecuación: B = B_sat * tanh(H / B_sat)
        - Calcula la inducción magnética B en función del campo magnético H.
        - Usa la función tanh para modelar la saturación del material.
        """
        B_sat = self.B_saturacion(masa)
        return B_sat * math.tanh(H / B_sat)
