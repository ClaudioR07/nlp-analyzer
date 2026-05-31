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

Para garantizar que los modelos de lenguaje pesado funcionen correctamente, se recomienda el uso de un entorno virtual (Anaconda o venv).

### 1. Clonar el repositorio y preparar el entorno
```bash
git clone [https://github.com/TU_USUARIO/nlp-analyzer.git](https://github.com/TU_USUARIO/nlp-analyzer.git)
cd nlp-analyzer
conda create -n pipeline_turismo python=3.9
conda activate pipeline_turismo