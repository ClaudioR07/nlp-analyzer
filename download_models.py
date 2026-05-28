"""
descarga todos los modelos lingüísticos necesarios
"""
import os
import sys
import subprocess

modelos_spacy = ['es_core_news_sm', 'en_core_web_sm', 'fr_core_news_sm']
for modelo in modelos_spacy:
    print(f"\nDescargando modelo de spaCy: {modelo}...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", modelo])
        print(f"{modelo} descargado con éxito.")
    except Exception as e:
        print(f"Error al descargar {modelo}: {e}")

# Transformers (pysentimiento y embeddings)
print("\nDescargando modelos de Deep Learning para Sentimientos y Tópicos...")
try:
    from pysentimiento import create_analyzer
    print("Descargando modelo de sentimientos en espanol...")
    create_analyzer(task="sentiment", lang="es")
    print("Descargando modelo de sentimientos en ingles...")
    create_analyzer(task="sentiment", lang="en")
    
    # Forzamos a sentence-transformers (usado por BERTopic) a descargar el modelo multilingüe
    from sentence_transformers import SentenceTransformer
    print("Descargando modelo de embeddings multilingüe para BERTopic...")
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("\n[ÉXITO] Todos los modelos se han descargado y guardado localmente.")

except Exception as e:
    print(f"\nProblema al precargar los modelos de Transformers: {e}")