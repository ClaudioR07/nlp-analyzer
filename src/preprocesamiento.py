import re
import spacy

def cargar_modelo_spacy(idioma):
    modelos = {'es': 'es_core_news_sm', 'en': 'en_core_web_sm', 'fr': 'fr_core_news_sm'}
    nombre_modelo = modelos.get(idioma, 'es_core_news_sm')
    try:
        return spacy.load(nombre_modelo)
    except IOError:
        raise IOError(f"Modelo {nombre_modelo} no instalado. Ejecute download_models.py")

def limpiar_comentario(texto, nlp):
    if not isinstance(texto, str) or texto.strip() == "":
        return ""
    texto = texto.lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    doc = nlp(texto)
    
    # tokenización, stopwords y lematización
    tokens_limpios = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and not token.is_space]
    return " ".join(tokens_limpios)

def preprocesar_dataframe(df, columna_texto, nlp):
    print(f"  Iniciando la limpieza de texto en la columna '{columna_texto}'...")
    df_resultado = df.copy()
    df_resultado['texto_limpio'] = df_resultado[columna_texto].apply(lambda x: limpiar_comentario(x, nlp))
    print("limpieza y lematización completadas")
    return df_resultado