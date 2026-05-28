
import os
import sys
import subprocess

print("=== INICIANDO DESCARGA DE MODELOS PARA EJECUCIÓN OFFLINE ===")

# spaCy
urls_spacy = [
    "https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0.tar.gz",
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz",
    "https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.7.0/fr_core_news_sm-3.7.0.tar.gz"
]

print("\nDescargando e instalando modelos de spaCy de forma directa...")
for url in urls_spacy:
    # Extraer el nombre del modelo de la URL para imprimirlo bonito
    nombre_modelo = url.split("/")[-1].split("-3.7.0")[0]
    print(f"\nProcesando: {nombre_modelo}...")
    try:
        # Aquí forzamos pip install directo desde la URL
        subprocess.check_call([sys.executable, "-m", "pip", "install", url])
        print(f"-> {nombre_modelo} instalado con éxito.")
    except Exception as e:
        print(f"[X] Error al instalar {nombre_modelo}: {e}")
        
# Sentimientos y Tópicos
print("\nDescargando modelos de Deep Learning para Sentimientos y Tópicos...")
try:
    from pysentimiento import create_analyzer
    print("Descargando analizador de sentimientos en Español...")
    create_analyzer(task="sentiment", lang="es")
    print("Descargando analizador de sentimientos en Inglés...")
    create_analyzer(task="sentiment", lang="en")
    
    from sentence_transformers import SentenceTransformer
    modelo_embeddings = 'paraphrase-multilingual-MiniLM-L12-v2'
    print(f"Descargando modelo de embeddings multilingüe ({modelo_embeddings})...")
    SentenceTransformer(modelo_embeddings)
    
    print("\n[ÉXITO] Todos los modelos se han guardado localmente para uso offline.")

except Exception as e:
    print(f"\n[ERROR] Hubo un problema al precargar los modelos: {e}")