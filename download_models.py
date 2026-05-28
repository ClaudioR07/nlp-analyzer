
import os
import sys
import subprocess

print("=== INICIANDO DESCARGA DE MODELOS PARA EJECUCIÓN OFFLINE ===")

# spaCy
modelos_spacy = ['es_core_news_sm', 'en_core_web_sm', 'fr_core_news_sm']
for modelo in modelos_spacy:
    print(f"\nDescargando modelo de spaCy: {modelo}...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", modelo])
        print(f"-> {modelo} descargado con éxito.")
    except Exception as e:
        print(f"Error al descargar {modelo}: {e}")

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