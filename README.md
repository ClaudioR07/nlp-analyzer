# 🌴 Analizador de Comentarios Turísticos (NLP Pipeline)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-orange.svg)
![spaCy](https://img.shields.io/badge/spaCy-NLP-success.svg)
![BERTopic](https://img.shields.io/badge/BERTopic-Modeling-yellow.svg)
![Offline](https://img.shields.io/badge/Status-100%25_Offline-brightgreen.svg)

Este proyecto es un **Pipeline de Procesamiento de Lenguaje Natural (NLP)** diseñado para procesar, limpiar y analizar masivamente opiniones turísticas. La arquitectura orquesta modelos de Inteligencia Artificial para entregar un Dashboard interactivo en HTML que funciona **100% sin conexión a internet**.

---

## ✨ Características Principales
- **Soporte Multilingüe:** Procesamiento adaptado para Español (`es`), Inglés (`en`) y Francés (`fr`).
- **Análisis de Sentimientos:** Clasificación neuronal de opiniones (POS, NEG, NEU).
- **Detección de Anomalías (Outliers):** Aislamiento de comentarios atípicos mediante `IsolationForest` y extracción de sus n-gramas principales.
- **Modelado Semántico (BERTopic):** Agrupación automatizada de comentarios en tópicos clave (Servicio, Infraestructura, Naturaleza).
- **Dashboard Offline (Plotly):** Reporte visual interactivo en un solo archivo HTML, exportable y ejecutable en cualquier navegador sin dependencias web.

---

## 🚀 Requisitos Previos e Instalación

Para garantizar que los modelos de lenguaje pesado funcionen correctamente, es obligatorio el uso de un entorno virtual (Anaconda o venv).

### 1. Clonar el repositorio y preparar el entorno
```bash
git clone [https://github.com/icaleb-dot/nlp-analyzer.git](https://github.com/icaleb-dot/nlp-analyzer.git)
cd nlp-analyzer
conda create -n pipeline_turismo python=3.9
conda activate pipeline_turismo
```

### 2. Instalar dependencias
Instala todas las librerías necesarias ejecutando:
```bash
pip install -r requirements.txt
```

### 3. Descargar modelos de IA (Primera Ejecución - Requiere Internet)
El pipeline depende de diccionarios de *spaCy* y modelos pre-entrenados de *HuggingFace*. Ejecuta el script automatizado para descargar los "cerebros" necesarios de forma local en tu computadora:
```bash
python download_models.py
```

---

## 💻 Guía de Ejecución (Paso a Paso)

Una vez configurado el entorno, todo el análisis se ejecuta a través del archivo `main.py`. 

### Comando Base
Asegúrate de estar en la carpeta raíz del proyecto y tener tu entorno activado. Ejecuta el orquestador con la siguiente estructura:

```bash
python main.py --input data/TU_ARCHIVO.csv --columna NOMBRE_COLUMNA --idioma es --titulo "TU TITULO" --paleta viridis
```

### 🌟 Ejemplo Práctico (Copia y pega para probar)
Si tienes un archivo llamado `riviera_nayarit_test.csv` en la carpeta `data/` y la columna que contiene las opiniones se llama `Comentario`, el comando sería:

```bash
python main.py --input data/riviera_nayarit_test.csv --columna Comentario --idioma es --titulo "Análisis Turístico - Riviera Nayarit" --paleta viridis
```

### ⚠️ Solución a problemas comunes:
* **Error de Codificación (UTF-8):** Si tu archivo CSV fue exportado desde Excel en Windows, es probable que marque error al leer acentos o la letra "ñ". Soluciónalo agregando el parámetro `-e latin-1` al final de tu comando:
  ```bash
  python main.py --input data/archivo.csv --columna Comentario --idioma es --titulo "Reporte" --paleta viridis -e latin-1
  ```
* **Offline Guarantee:** La primera vez que se ejecuta el comando `main.py`, se validarán los modelos en caché. A partir del segundo uso, puedes apagar tu conexión a internet y el script se ejecutará 100% de forma local.

---

## ⚙️ Tabla de Parámetros del CLI

| Parámetro | Requerido | Descripción | Ejemplo |
| :--- | :---: | :--- | :--- |
| `-i, --input` | **Sí** | Ruta del archivo CSV a analizar. | `data/playas.csv` |
| `-c, --columna` | **Sí** | Nombre exacto de la columna con los textos. | `Comentario` |
| `-l, --idioma` | **Sí** | Idioma de los datos (`es`, `en`, `fr`). | `es` |
| `-t, --titulo` | **Sí** | Título que aparecerá en el Dashboard HTML. | `"Análisis 2026"` |
| `-p, --paleta` | **Sí** | Paleta de colores (`viridis`, `cividis`, `plasma`, `inferno`). | `viridis` |
| `-e, --encoding`| No | Codificación del archivo (Útil para CSVs de Excel). | `latin-1` o `utf-8` |

---

## 🏗️ Arquitectura y Roles del Equipo

El proyecto está modularizado en 4 etapas principales, integradas como un solo flujo continuo:

* 🧹 **Integrante 1 (Preprocesamiento Léxico y CLI):** * Orquestación de argumentos por consola (`argparse`).
  * Limpieza estandarizada de texto, lematización y tokenización utilizando `spaCy`.
* 📊 **Integrante 2 (Outliers y Sentimientos):** * Identificación de comentarios atípicos mediante el algoritmo `IsolationForest` de *Scikit-Learn*.
  * Clasificación de polaridad utilizando redes neuronales de `pysentimiento` y `TextBlob`.
* 🧠 **Integrante 3 (Modelado de Tópicos y Semántica):** * Extracción de tópicos guiados utilizando `BERTopic`.
  * Cálculo matemático de "Similitud de Coseno" con `sentence-transformers` para medir la relevancia económica ("Precio/Valor") de cada comentario.
* 🧩 **Integrante 4 (Orquestación general y Visualización):** * Fusión de los módulos en `main.py` para asegurar un flujo de datos continuo y resolución de formatos matriciales (CSR a arrays densos).
  * Renderizado del módulo interactivo con `Plotly`, integrando las variables de todo el equipo en un único dashboard web escalable y completamente independiente (`reporte_turismo_offline.html`).

---

## 📈 Resultados Finales
Al finalizar la ejecución en la terminal, se generará de manera automática el archivo `reporte_turismo_offline.html` en el directorio raíz. 

Este Dashboard es de página única y contiene visualizaciones interactivas de grado profesional:
1. **Gráfico de Dona:** Proporción y distribución general de sentimientos.
2. **Treemap:** Nube de palabras interactiva con los términos globales más frecuentes.
3. **Gráfico de Barras (Outliers):** Extracción de los n-gramas más repetidos exclusivamente en comentarios atípicos.
4. **Sunburst Chart:** Jerarquía interactiva navegable de Sentimientos y sus respectivos Tópicos semánticos.
5. **Scatter & Violin Plots:** Análisis tridimensional de percepción de valor económico aglomerado por densidad y sentimiento.
