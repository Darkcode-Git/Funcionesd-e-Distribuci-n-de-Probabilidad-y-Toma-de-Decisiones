from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, chisquare


@dataclass(frozen=True)
class ConfiguracionBinomial:
    """Estructura para almacenar los datos del problema."""

    ensamblajes: int = 100
    intentos_por_ensamblaje: int = 15
    probabilidad_exito: float = 0.8
    semilla: int = 42
    alfa: float = 0.05


def combinar_categorias_para_chi2(
    observados: np.ndarray, esperados: np.ndarray, minimo_esperado: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combina categorías adyacentes hasta que cada frecuencia esperada sea >= minimo_esperado.
    Esto mejora la validez de la aproximación de chi-cuadrado.
    """
    grupos_obs = []
    grupos_exp = []
    acumulado_obs = 0.0
    acumulado_exp = 0.0

    for obs, exp in zip(observados, esperados):
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


def main() -> None:
    cfg = ConfiguracionBinomial()
    np.random.seed(cfg.semilla)

    # 1) Simulación de ensamblajes
    exitos = np.random.binomial(
        n=cfg.intentos_por_ensamblaje,
        p=cfg.probabilidad_exito,
        size=cfg.ensamblajes,
    )

    promedio_muestral = np.mean(exitos)
    varianza_muestral = np.var(exitos, ddof=1)
    varianza_teorica = cfg.intentos_por_ensamblaje * cfg.probabilidad_exito * (
        1 - cfg.probabilidad_exito
    )

    print("=== Numeral 1: Simulación de 100 ensamblajes ===")
    print(f"Promedio muestral de éxitos: {promedio_muestral:.4f}")
    print(f"Varianza muestral de éxitos: {varianza_muestral:.4f}")
    print(f"Varianza teórica n·p·(1-p):  {varianza_teorica:.4f}")
    print(
        "Interpretación: la varianza muestral debe acercarse a la teórica; "
        "la diferencia observada se explica por variabilidad aleatoria."
    )

    # 2) Probabilidades solicitadas
    n = cfg.intentos_por_ensamblaje
    p = cfg.probabilidad_exito
    prob_exactamente_12 = binom.pmf(12, n, p)
    prob_al_menos_10 = 1 - binom.cdf(9, n, p)
    prob_entre_8_y_12 = binom.cdf(12, n, p) - binom.cdf(7, n, p)

    print("\n=== Numeral 2: Probabilidades binomiales (n=15, p=0.8) ===")
    print(f"a) P(X = 12):               {prob_exactamente_12:.6f}")
    print(f"b) P(X >= 10):              {prob_al_menos_10:.6f}")
    print(f"c) P(8 <= X <= 12):         {prob_entre_8_y_12:.6f}")

    # 3) PMF teórica vs histograma simulado
    x = np.arange(0, n + 1)
    pmf_teorica = binom.pmf(x, n, p)
    frecuencias_observadas = np.bincount(exitos, minlength=n + 1)
    frecuencias_relativas = frecuencias_observadas / cfg.ensamblajes

    plt.figure(figsize=(10, 5))
    ancho = 0.4
    plt.bar(x - ancho / 2, pmf_teorica, width=ancho, label="PMF teórica")
    plt.bar(
        x + ancho / 2,
        frecuencias_relativas,
        width=ancho,
        alpha=0.8,
        label="Frecuencia relativa simulada",
    )
    plt.title("Distribución Binomial: Teórica vs Simulada")
    plt.xlabel("Número de éxitos en 15 intentos")
    plt.ylabel("Probabilidad / Frecuencia relativa")
    plt.xticks(x)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pmf_vs_simulacion_binomial.png", dpi=150)
    plt.close()

    print("\n=== Numeral 3: Comparación gráfica ===")
    print(
        "Se generó 'pmf_vs_simulacion_binomial.png'. "
        "Visualmente, las frecuencias simuladas deberían aproximar la PMF teórica."
    )

    # 4) Prueba de bondad de ajuste (chi-cuadrado)
    esperados = cfg.ensamblajes * pmf_teorica
    obs_agrupados, exp_agrupados = combinar_categorias_para_chi2(
        frecuencias_observadas, esperados, minimo_esperado=5.0
    )
    estadistico, p_valor = chisquare(f_obs=obs_agrupados, f_exp=exp_agrupados)

    print("\n=== Numeral 4: Prueba de bondad de ajuste Chi-cuadrado ===")
    print(f"Estadístico Chi-cuadrado: {estadistico:.4f}")
    print(f"Valor p:                  {p_valor:.6f}")
    if p_valor > cfg.alfa:
        print(
            f"Conclusión (alfa={cfg.alfa}): no se rechaza H0; "
            "los datos simulados son consistentes con Binomial(15, 0.8)."
        )
    else:
        print(
            f"Conclusión (alfa={cfg.alfa}): se rechaza H0; "
            "los datos simulados no son consistentes con Binomial(15, 0.8)."
        )


if __name__ == "__main__":
    main()
