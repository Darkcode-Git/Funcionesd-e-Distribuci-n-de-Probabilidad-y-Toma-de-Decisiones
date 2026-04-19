import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, chisquare


def agrupar_categorias(frec_obs, frec_exp, minimo_esperado=5):
    """
    Agrupa categorías contiguas hasta alcanzar una frecuencia esperada mínima.

    Args:
        frec_obs: Frecuencias observadas por categoría.
        frec_exp: Frecuencias esperadas por categoría.
        minimo_esperado: Umbral mínimo recomendado para prueba chi-cuadrado.

    Returns:
        Tuple[np.ndarray, np.ndarray]: frecuencias observadas y esperadas agrupadas.
    """
    grupos_obs = []
    grupos_exp = []

    acumulado_obs = 0.0
    acumulado_exp = 0.0

    for obs, exp in zip(frec_obs, frec_exp):
        acumulado_obs += obs
        acumulado_exp += exp
        if acumulado_exp >= minimo_esperado:
            grupos_obs.append(acumulado_obs)
            grupos_exp.append(acumulado_exp)
            acumulado_obs = 0.0
            acumulado_exp = 0.0

    if acumulado_exp > 0:
        if grupos_exp:
            grupos_obs[-1] += acumulado_obs
            grupos_exp[-1] += acumulado_exp
        else:
            grupos_obs.append(acumulado_obs)
            grupos_exp.append(acumulado_exp)

    return np.array(grupos_obs), np.array(grupos_exp)


def main():
    # Parámetros iniciales
    n = 15
    p = 0.8
    num_simulaciones = 100
    np.random.seed(42)

    # 1. Generar datos simulados
    datos_simulados = np.random.binomial(n, p, num_simulaciones)

    # Cálculo de promedio y varianza de los datos simulados
    promedio_sim = np.mean(datos_simulados)
    # Se usa varianza poblacional (ddof=0) para comparar con la varianza teórica n*p*(1-p).
    varianza_sim = np.var(datos_simulados)
    varianza_teorica = n * p * (1 - p)

    print("1) Simulación y propiedades")
    print(f"Promedio simulado: {promedio_sim:.4f}")
    print(f"Media teórica (n*p): {n * p:.4f}")
    print(f"Varianza simulada: {varianza_sim:.4f}")
    print(f"Varianza teórica (n*p*q): {varianza_teorica:.4f}")
    print()

    # 2. Cálculos de probabilidad binomial
    prob_exacta_12 = binom.pmf(12, n, p)
    prob_al_menos_10 = 1 - binom.cdf(9, n, p)
    prob_rango_8_12 = binom.cdf(12, n, p) - binom.cdf(7, n, p)

    print("2) Probabilidades binomiales")
    print(f"a) P(X=12): {prob_exacta_12:.4f}")
    print(f"b) P(X>=10): {prob_al_menos_10:.4f}")
    print(f"c) P(8<=X<=12): {prob_rango_8_12:.4f}")
    print()

    # 3. Visualización y comparación
    x = np.arange(0, n + 1)
    pmf_teorica = binom.pmf(x, n, p)

    plt.figure(figsize=(10, 6))
    plt.hist(
        datos_simulados,
        bins=np.arange(n + 2) - 0.5,
        density=True,
        alpha=0.6,
        color="skyblue",
        label="Datos simulados (histograma)",
    )
    plt.bar(
        x,
        pmf_teorica,
        alpha=0.4,
        color="orange",
        label="Binomial teórica (PMF)",
    )

    plt.title("Comparación: Datos simulados vs. Binomial teórica")
    plt.xlabel("Número de éxitos")
    plt.ylabel("Probabilidad / Frecuencia relativa")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("comparacion_binomial.png", dpi=120)
    plt.close()

    # 4. Prueba de bondad de ajuste chi-cuadrado
    frec_observada, _ = np.histogram(datos_simulados, bins=np.arange(n + 2) - 0.5)
    frec_esperada = pmf_teorica * num_simulaciones

    obs_agrupada, esp_agrupada = agrupar_categorias(frec_observada, frec_esperada, minimo_esperado=5)

    # Reescalado para igualar sumas y evitar error numérico en chisquare.
    esp_agrupada = esp_agrupada * (obs_agrupada.sum() / esp_agrupada.sum())

    stat, p_value = chisquare(obs_agrupada, f_exp=esp_agrupada)

    print("3) Visualización")
    print("Se generó el archivo: comparacion_binomial.png")
    print()
    print("4) Prueba de bondad de ajuste (Chi-cuadrado)")
    print(f"Estadístico Chi-cuadrado: {stat:.4f}")
    print(f"Valor p obtenido: {p_value:.4f}")
    if p_value > 0.05:
        print("Conclusión: no se rechaza H0; los datos son consistentes con Binomial(15, 0.8).")
    else:
        print("Conclusión: se rechaza H0; existe evidencia de discrepancia con Binomial(15, 0.8).")


if __name__ == "__main__":
    main()
