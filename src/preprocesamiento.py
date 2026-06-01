import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from collections import Counter
import nltk

nltk.download('punkt_tab', quiet=True)

def cargar_modelo_spacy(idioma):
    modelos = {
        'es': 'es_core_news_sm',
        'en': 'en_core_web_sm',
        'fr': 'fr_core_news_sm'
    }
    nombre_modelo = modelos.get(idioma, 'es_core_news_sm')
    try:
        return spacy.load(nombre_modelo)
    except IOError:
        raise IOError(f"Modelo {nombre_modelo} no instalado. Ejecute download_models.py")

def limpiar_comentario(texto, nlp):
    if not isinstance(texto, str) or texto.strip() == "":
        return ""

    texto = texto.lower()

    texto = re.sub(r'[áàäâ]', 'a', texto)
    texto = re.sub(r'[éèëê]', 'e', texto)
    texto = re.sub(r'[íìïî]', 'i', texto)
    texto = re.sub(r'[óòöô]', 'o', texto)
    texto = re.sub(r'[úùüû]', 'u', texto)

    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)

    doc = nlp(texto)

    tokens_limpios = [
    token.text
    for token in doc
    if (
        not token.is_stop
        and token.is_alpha
        and not token.is_space
        and len(token.text) > 2
    )
]

    return " ".join(tokens_limpios)

def preprocesar_dataframe(df, columna_texto, nlp):
    print(f"  Iniciando la limpieza de texto en la columna '{columna_texto}'...")
    df_resultado = df.copy()
    df_resultado['texto_limpio'] = df_resultado[columna_texto].apply(
        lambda x: limpiar_comentario(x, nlp)
    )
    print("  Limpieza y lematización completadas.")
    return df_resultado

def detectar_outliers(textos_limpios: list, contaminacion: float = 0.05):
    """
    Detecta comentarios atípicos usando Isolation Forest sobre TF-IDF.
    """
    print(f"  Detectando outliers (contaminación={contaminacion})...")

    # Filtrar textos vacíos antes de vectorizar
    textos_validos = [t if t.strip() != "" else "vacio" for t in textos_limpios]

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(textos_validos).toarray()

    modelo = IsolationForest(contamination=contaminacion, random_state=42)
    etiquetas = modelo.fit_predict(X)  # -1 = outlier, 1 = normal

    indices_outliers = [i for i, e in enumerate(etiquetas) if e == -1]
    indices_normales = [i for i, e in enumerate(etiquetas) if e == 1]

    print(f"  Outliers detectados: {len(indices_outliers)} | Normales: {len(indices_normales)}")
    return indices_outliers, indices_normales


def _generar_ngramas(textos: list, n: int, top_k: int = 15):
    """Genera los top_k n-gramas más frecuentes de una lista de textos."""
    from nltk.util import ngrams as nltk_ngrams

    todos = []
    for texto in textos:
        tokens = texto.split()
        if len(tokens) >= n:
            todos.extend(nltk_ngrams(tokens, n))

    if not todos:
        return []

    conteo = Counter(todos)
    # Convertir tuplas a strings legibles: ("buen", "hotel") → "buen hotel"
    return [(" ".join(ngrama), freq) for ngrama, freq in conteo.most_common(top_k)]


def analizar_outliers_ngramas(textos_outliers: list, top_k: int = 15) -> dict:
    """
    Genera unigramas, bigramas y trigramas de los comentarios outliers.

    Parámetros:
        textos_outliers : lista de strings (ya limpios) etiquetados como atípicos
        top_k           : cuántos n-gramas mostrar por tipo

    Retorna:
        dict con claves 'unigramas', 'bigramas', 'trigramas',
        cada una con lista de (ngrama_str, frecuencia)
    """
    print(f"  Analizando n-gramas de {len(textos_outliers)} comentarios outliers...")

    resultado = {
        "unigramas": _generar_ngramas(textos_outliers, 1, top_k),
        "bigramas":  _generar_ngramas(textos_outliers, 2, top_k),
        "trigramas": _generar_ngramas(textos_outliers, 3, top_k),
    }

    return resultado

