# Analizador de Comentarios Turísticos (NLP Pipeline)

Este proyecto ejecuta un pipeline completo offline para analizar opiniones turísticas, integrando limpieza léxica, detección de outliers, análisis de sentimientos, modelado de tópicos y generación de dashboards interactivos.

## Requisitos Previos e Instalación
Instrucciones para configurar el entorno virtual e instalar las dependencias (`pip install -r requirements.txt`).

## Ejecución del Script (CLI)
Para ejecutar el orquestador, utiliza el siguiente formato:
`python main.py --csv data/archivo.csv --columna texto --idioma es --titulo "Reporte" --paleta viridis`

## Arquitectura y Roles
* **Integrante 1:** CLI y Preprocesamiento Léxico.
* **Integrante 2:** Outliers y Sentimientos (pysentimiento).
* **Integrante 3:** Tópicos (BERTopic) y Similitud "Precio".
* **Integrante 4:** Integración, Orquestación y Visualización Interactiva Plotly.