import re
import spacy

def cargar_modelo_spacy(idioma):

    modelos = {
        'es': 'es_core_news_sm',
        'en': 'en_core_web_sm',
        'fr': 'fr_core_news_sm'
    }
    
    nombre_modelo = modelos.get(idioma, 'es_core_news_sm')
    
    try:
        nlp = spacy.load(nombre_modelo)
    except IOError:
        raise IOError(
            f"El modelo de spaCy '{nombre_modelo}' no está instalado en este entorno.\n")
    return nlp


def limpiar_comentario(texto, nlp):

    if not isinstance(texto, str) or texto.strip() == "":
        return ""

    texto = texto.lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    doc = nlp(texto)

    tokens_limpios = []
    for token in doc:
        if token.is_stop:
            continue
        if not token.is_alpha:
            continue
        if token.is_space:
            continue

        # lematización 
        tokens_limpios.append(token.lemma_)

    # comentario limpio como una sola cadena de texto unida por espacios
    return " ".join(tokens_limpios)


def preprocesar_dataframe(df, columna_texto, nlp):
    print(f"limpieza de texto en la columna '{columna_texto}'...")
    df_resultado = df.copy()
    
    # Aplicamos la función fila por fila y guardamos el resultado en una nueva columna
    df_resultado['texto_limpio'] = df_resultado[columna_texto].apply(lambda x: limpiar_comentario(x, nlp))
    
    print("limpieza y lematización lista!")
    return df_resultado