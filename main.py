import pandas as pd
import numpy as np
from src.cli import obtener_argumentos
from src.preprocesamiento import cargar_modelo_spacy, preprocesar_dataframe
from src.sentimientos import AnalizadorSentimientosPipeline
from src.topicos import AnalizadorSemantico
from src.visualizaciones import generar_graficas

def main():
    # validación del CLI (Tu Módulo)
    args = obtener_argumentos()
    
    print(f"\n[1/5] Leyendo el archivo de datos: {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error crítico al abrir el archivo CSV: {e}")
        return

    # preprocesamiento de texto (Tu Módulo)
    print("\n[2/5] Cargando pipeline de spaCy y limpiando texto...")
    nlp = cargar_modelo_spacy(args.idioma)
    df_limpio = preprocesar_dataframe(df, args.columna, nlp)

    # 3. Clasificación de Sentimientos (Módulo Integrante 2 - Adaptado a TextBlob/Pysentimiento)
    print("\n[3/5] Ejecutando análisis de sentimientos en lote...")
    lector_sentimientos = AnalizadorSentimientosPipeline(args.idioma)
    df_limpio['Polaridad_Clase'] = df_limpio[args.columna].apply(lambda x: lector_sentimientos.analizar(x))
    
    # Mapeo numérico para las gráficas del Integrante 4
    mapeo_numerico = {'POS': 1.0, 'NEU': 0.0, 'NEG': -1.0}
    df_limpio['Polaridad_Sentimiento'] = df_limpio['Polaridad_Clase'].map(mapeo_numerico)

    # Separar las particiones solicitadas en la rúbrica
    df_positivos = df_limpio[df_limpio['Polaridad_Clase'] == 'POS'].copy()
    df_negativos = df_limpio[df_limpio['Polaridad_Clase'] == 'NEG'].copy()

    # 4. Modelado de Tópicos e Inteligencia Semántica (Módulo Integrante 3 - ¡Exacto a su código!)
    print("\n[4/5] Iniciando modelado de tópicos semánticos por particiones...")
    # Inicializamos su clase con el idioma y el umbral de 30
    analizador_semantico = AnalizadorSemantico(idioma=args.idioma, umbral_minimo=30)
    
    # Llamamos a su método estrella que decide si usa BERTopic o Frecuencias
    res_pos = analizador_semantico.procesar_particion(df_positivos, col_limpia='texto_limpio', col_original=args.columna)
    res_neg = analizador_semantico.procesar_particion(df_negativos, col_limpia='texto_limpio', col_original=args.columna)
    
    # Mostramos en consola los resultados si se ejecutó BERTopic con éxito
    if res_pos['metodo'] == 'bertopic':
        print("\n--> Tópicos Detectados en Comentarios POSITIVOS:")
        for t in res_pos['resumen']:
            print(f"  * Tópico {t['id_topico']} | Palabras Clave: {t['palabras_clave']}")
            print(f"    Comentario representativo: \"{t['comentario_representative']}\"")
            
    if res_neg['metodo'] == 'bertopic':
        print("\n--> Tópicos Detectados en Comentarios NEGATIVOS:")
        for t in res_neg['resumen']:
            print(f"  * Tópico {t['id_topico']} | Palabras Clave: {t['palabras_clave']}")
            print(f"    Comentario representativo: \"{t['comentario_representative']}\"")

    # Si se fue por frecuencias por no alcanzar el umbral, mostramos el top de palabras
    if res_pos['metodo'] == 'frecuencia':
        print(f"\n--> Partición positiva pequeña. Top palabras frecuentes: {res_pos['datos']}")
    if res_neg['metodo'] == 'frecuencia':
        print(f"\n--> Partición negativa pequeña. Top palabras frecuentes: {res_neg['datos']}")

    # Ejecutamos su función de similitud de coseno para Precio/Valor
    top_precios = analizador_semantico.analizar_precio_valor(df_limpio, col_limpia='texto_limpio', col_original=args.columna)
    
    print("\n=========================================")
    print("Top 5 Comentarios sobre Precio/Valor:")
    print("=========================================")
    for item in top_precios:
        print(f"Similitud {item['similitud']:.4f}: {item['comentario_original']}")
    print("=========================================\n")

    # --- PREPARACIÓN DE DATOS PARA LAS GRÁFICAS DEL INTEGRANTE 4 ---
    # Mapeamos los tópicos asignados de vuelta al DataFrame original para el Scatter Plot
    df_limpio['Tópico'] = 'Outlier / Otros'
    
    if res_pos['metodo'] == 'bertopic' and not df_positivos.empty:
        df_limpio.loc[df_limpio['Polaridad_Clase'] == 'POS', 'Tópico'] = [f"Pos-Top {t}" for t in res_pos['topicos']]
    elif res_pos['metodo'] == 'frecuencia' and not df_positivos.empty:
        df_limpio.loc[df_limpio['Polaridad_Clase'] == 'POS', 'Tópico'] = 'Baja Frecuencia (Positivos)'

    if res_neg['metodo'] == 'bertopic' and not df_negativos.empty:
        df_limpio.loc[df_limpio['Polaridad_Clase'] == 'NEG', 'Tópico'] = [f"Neg-Top {t}" for t in res_neg['topicos']]
    elif res_neg['metodo'] == 'frecuencia' and not df_negativos.empty:
        df_limpio.loc[df_limpio['Polaridad_Clase'] == 'NEG', 'Tópico'] = 'Baja Frecuencia (Negativos)'

    # Generamos la columna de Similitud_Precio requerida por la gráfica del Integrante 4
    # Reutilizamos el modelo de embeddings de tu compañero para calcular el vector completo fila por fila
    frase_ref = "The price is fair" if args.idioma == 'en' else "Le prix est juste" if args.idioma == 'fr' else "El precio es justo"
    vector_ref = analizador_semantico.embedding_model.encode([frase_ref])
    vectores_corp = analizador_semantico.embedding_model.encode(df_limpio['texto_limpio'].tolist(), show_progress_bar=False)
    
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    df_limpio['Similitud_Precio'] = cos_sim(vector_ref, vectores_corp)[0]
    df_limpio['Comentario_Original'] = df_limpio[args.columna]

    # 5. Generación de Visualizaciones Interactivas (Módulo Integrante 4 - ¡Exacto a su código!)
    print("\n[5/5] Renderizando reportes visuales en HTML con Plotly...")
    generar_graficas(
        datos=df_limpio, 
        paleta=args.paleta, 
        titulo=args.titulo, 
        modo_oscuro=True
    )
    print("\n=== ¡EJECUCIÓN DEL PIPELINE COMPLETADA CON ÉXITO! ===")

if __name__ == '__main__':
    main()