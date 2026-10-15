# Model Card — FinanceGuard Churn v0.1

## Uso previsto

Priorizar clientes para una campaña hipotética de retención y demostrar un flujo reproducible de ciencia de datos. No debe usarse para negar productos, fijar precios, evaluar solvencia ni tomar decisiones automatizadas sobre personas.

## Datos

- 10.000 registros de clientes.
- Objetivo: `Exited` (20,37% positivos).
- Predictores demográficos y de relación bancaria agregada.
- Identificadores y apellido excluidos.
- Sin fechas, historial transaccional, campañas previas ni variable de respuesta causal.

## Protocolo de evaluación

Se reserva un 20% estratificado como conjunto de test. En el 80% de desarrollo se comparan regresión logística e HistGradientBoosting mediante validación cruzada estratificada de cinco folds. La selección utiliza PR-AUC como métrica principal.

El umbral de decisión se determina con probabilidades out-of-fold generadas exclusivamente sobre desarrollo. Puede seleccionarse el máximo beneficio esperado sin restricciones o incorporar un límite operativo sobre la proporción de clientes contactados. El modelo ganador se ajusta sobre todo el conjunto de desarrollo y se evalúa sobre test.

## Resultado de referencia

El modelo seleccionado fue HistGradientBoosting, con un PR-AUC medio de 0,7004 durante la validación cruzada.

Para el escenario operativo que limita los contactos al 20% de la cartera, se seleccionó un umbral de 0,325.

| Métrica | Test |
|---|---:|
| ROC-AUC | 0,8704 |
| PR-AUC | 0,7196 |
| Brier score | 0,0992 |
| Accuracy | 0,8530 |
| Precision | 0,6423 |
| Recall | 0,6265 |
| F1 | 0,6343 |
| Lift al 10% | 4,1278 |
| Tasa de contacto | 19,85% |

Los supuestos económicos son: costo de contacto de 10 unidades monetarias, valor de un cliente retenido de 250 y efectividad hipotética de campaña del 30%.

La política sin restricción maximiza el beneficio esperado con un umbral de 0,15 y requiere contactar aproximadamente al 38% de la cartera. La política limitada al 20% utiliza un umbral de 0,325 y conserva aproximadamente el 85% del beneficio simulado reduciendo los contactos cerca de un 47%.

Estos valores representan una simulación de escenarios. La efectividad de la campaña no fue observada en el dataset y no constituye una estimación causal.

## Limitaciones y riesgos

- Los datos son estáticos y no permiten validación temporal ni medición de deriva.
- El origen y proceso de muestreo no están suficientemente documentados.
- `Gender` y `Geography` requieren revisión de equidad antes de cualquier uso real.
- El modelo predice asociación con abandono, no el efecto causal de una intervención.
- La utilidad monetaria es sensible a supuestos que deben reemplazarse con información de negocio.
- El desempeño fuera de esta muestra es desconocido.

## Controles requeridos antes de producción

Validación temporal, auditoría de calidad y representatividad, análisis de equidad, calibración por segmento, experimento controlado de campaña, monitoreo de datos y desempeño, aprobación humana y procedimiento de reversión.
