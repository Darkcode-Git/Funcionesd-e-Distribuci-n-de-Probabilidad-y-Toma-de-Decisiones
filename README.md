Funciones de Distribución de Probabilidad y Toma de Decisiones
Esta unidad  ofrece una introducción detallada al análisis multivariado de datos, una disciplina que examina simultáneamente múltiples variables para comprender fenómenos complejos. El material explica metodologías fundamentales como la correlación, la agrupación y el análisis de varianza, destacando su utilidad tanto en la investigación científica como en el ámbito profesional. Se presentan tres enfoques principales: relacionar variables, clasificar información y resumir datos para facilitar su interpretación. Además, el texto incluye un componente práctico mediante el uso del lenguaje R, enseñando a procesar bases de datos reales para identificar patrones y dependencias lineales. Finalmente, se subraya la importancia de conceptos como la covarianza y el planteamiento de hipótesis para resolver problemas basados en evidencias estadísticas.
<img width="2667" height="1487" alt="image" src="https://github.com/user-attachments/assets/828e5d05-8c9e-4ca0-b634-7b0305cd6027" />
<img width="2667" height="1488" alt="image" src="https://github.com/user-attachments/assets/b7283c16-fd85-4407-b7b7-51215009db6a" />
<img width="2667" height="1488" alt="image" src="https://github.com/user-attachments/assets/1aac8a08-a362-486b-95f0-bfc42a3a1b59" />
<img width="2667" height="1488" alt="image" src="https://github.com/user-attachments/assets/2cb7d331-fcb7-4a65-a13c-b71f908b8035" />
<img width="2667" height="1488" alt="image" src="https://github.com/user-attachments/assets/58c7f781-4710-4380-8b7f-8fea6e8f2d6b" />
<img width="2667" height="1488" alt="image" src="https://github.com/user-attachments/assets/fb4775b9-4b5d-4f54-b8f9-0bf45ef9e703" />
<img width="2667" height="1488" alt="image" src="https://github.com/user-attachments/assets/aab6f377-2107-443c-8e05-74fd831fd5c8" />

## Evidencia de aprendizaje: Distribución Binomial

Se agregó el archivo:

- `simulacion_binomial.py`

Este script resuelve la actividad de la Unidad 2:

- Simulación de 100 ensamblajes con 15 intentos y probabilidad de éxito de 0.8.
- Cálculo de promedio y varianza muestral y comparación con la varianza teórica `n * p * (1-p)`.
- Cálculo de probabilidades binomiales solicitadas.
- Comparación visual entre PMF teórica y frecuencias simuladas.
- Prueba de bondad de ajuste Chi-cuadrado con interpretación del valor p.

Ejecución:

```bash
pip install -r requirements.txt
python simulacion_binomial.py
```

El script genera la figura:

- `pmf_vs_simulacion_binomial.png`
