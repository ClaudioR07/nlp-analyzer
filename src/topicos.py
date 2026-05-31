import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
import warnings


warnings.filterwarnings("ignore")

class AnalizadorSemantico:
    def __init__(self, idioma='es', umbral_minimo=30):
        """
        Inicializa el analizador.
        """
        self.idioma = idioma
        self.umbral_minimo = umbral_minimo
        
        print("Cargando modelo de embeddings multilingüe (esto puede tardar la primera vez)...")
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        # Nota: Si se evalúan otros idiomas, estas palabras guían al modelo pero
        # los embeddings multilingües ayudarán a asociarlas a sus traducciones implícitas.
        self.seeded_topics = [
            # Tópico 0: Entorno y la infraestructura (lugar físico)
            ["infraestructura", "instalaciones", "baños", "mantenimiento", "estacionamiento", "ubicación", "limpieza", "seguridad"],
            
            # Tópico 1: Atractivos y experiencia (estético / actividades)
            ["experiencia", "atractivo", "paisaje", "vista", "ambiente", "actividades", "diversión", "naturaleza", "clima"],
            
            # Tópico 2: Servicio y hospitalidad
            ["servicio", "atención", "personal", "personal", "amabilidad", "comida", "bebida", "rapidez", "hospitalidad"]
        ]

    def procesar_particion(self, df_particion, col_limpia='texto_limpio', col_original='comentario'):
        """
        Recibe una partición (ej. solo positivos o solo negativos).
        Evalúa el umbral y decide si aplica BERTopic o Frecuencia de palabras.
        """
        textos_limpios = df_particion[col_limpia].tolist()
        textos_originales = df_particion[col_original].tolist()
        
        if len(textos_limpios) == 0:
            return {"metodo": "vacio", "datos": None}

        if len(textos_limpios) < self.umbral_minimo:
            print(f"  -> Umbral no alcanzado ({len(textos_limpios)} < {self.umbral_minimo}). Usando frecuencia de palabras...")
            return self._generar_frecuencias(textos_limpios)
        else:
            print(f"  -> Umbral alcanzado ({len(textos_limpios)}). Ejecutando BERTopic guiado...")
            return self._ejecutar_bertopic_guiado(textos_limpios, textos_originales)

    def _generar_frecuencias(self, textos_limpios: list):
        """
        Alternativa a BERTopic para grupos pequeños. Genera n-gramas o frecuencias.
        """
        # stopwords dependiendo del idioma
        stop_words_lang = 'english' if self.idioma == 'en' else None # nota: Sklearn no trae de 'es' por defecto, pero el texto ya viene limpio de spacy
        
        vectorizer = CountVectorizer(stop_words=stop_words_lang)
        try:
            X = vectorizer.fit_transform(textos_limpios)
            palabras = vectorizer.get_feature_names_out()
            frecuencias = X.sum(axis=0).A1
            
            diccionario_frecuencias = dict(zip(palabras, frecuencias))
            top_palabras = Counter(diccionario_frecuencias).most_common(15)
            
            return {
                "metodo": "frecuencia",
                "datos": top_palabras
            }
        except ValueError:
            # Por si todos los textos están vacíos después de limpiar
            return {"metodo": "frecuencia", "datos": []}

    def _ejecutar_bertopic_guiado(self, textos_limpios: list, textos_originales: list):
        """
        Implementa BERTopic con Guided Topic Modeling (seeded_topics).
        Extrae palabras clave y los comentarios más representativos (Punto 5).
        """
        # calcular embeddings previamente para mayor control
        embeddings = self.embedding_model.encode(textos_limpios, show_progress_bar=False)
        
        # BERTopic con tópicos ancla
        topic_model = BERTopic(seed_topic_list=self.seeded_topics, language="multilingual", nr_topics=15)
        topics, probs = topic_model.fit_transform(textos_limpios, embeddings)
        
        # información de los tópicos
        info_topicos = topic_model.get_topic_info()
        
        # palabras representativas y el comentario original más cercano al centroide
        resultados_topicos = []
        for topic_id in info_topicos['Topic']:
            if topic_id == -1:
                continue # Ignorar el tópico -1 (outliers de BERTopic)
                
            # Extraer palabras clave
            palabras_clave = [word for word, _ in topic_model.get_topic(topic_id)[:5]]
            
            # Buscar el documento (comentario) más representativo para este tópico
            # BERTopic guarda índices de documentos representativos internamente
            repr_docs = topic_model.get_representative_docs(topic_id)
            comentario_repr = repr_docs[0] if repr_docs else "No se encontró comentario representativo."
            
            # Intentar mapear el comentario limpio al original
            # (aproximación segura)
            comentario_original_repr = comentario_repr
            if comentario_repr in textos_limpios:
                idx = textos_limpios.index(comentario_repr)
                comentario_original_repr = textos_originales[idx]
            
            resultados_topicos.append({
                "id_topico": topic_id,
                "nombre_topico": info_topicos[info_topicos['Topic'] == topic_id]['Name'].values[0],
                "palabras_clave": palabras_clave,
                "comentario_representativo": comentario_original_repr
            })
            
        return {
            "metodo": "bertopic",
            "modelo": topic_model,
            "topicos": topics, # Asignación de cada comentario a un tópico (para el scatter plot)
            "resumen": resultados_topicos
        }

    def analizar_precio_valor(self, df, col_limpia='texto_limpio', col_original='comentario'):
        """
        Análisis del concepto 'Precio, valor, costo' usando similitud de coseno.
        """
        print("Realizando análisis semántico de 'Precio / Valor / Costo'...")
        textos_limpios = df[col_limpia].tolist()
        textos_originales = df[col_original].tolist()
        
        if len(textos_limpios) == 0:
            return []
            
        # Definimos una lista de frases que ataquen el tema desde diferentes ángulos
        if self.idioma == 'en':
            frases_referencia = [
                "The price is fair for the value and cost of the service",
                "It is cheap, expensive, or a good value for money",
                "Excellent quality for a reasonable price",
                "They charged me too much money for what it is"
            ]
        elif self.idioma == 'fr':
            frases_referencia = [
                "Le prix est juste pour la valeur et le coût du service",
                "C'est cher, pas cher, ou un bon rapport qualité prix",
                "Excellente qualité pour un prix raisonnable",
                "Ils m'ont demandé trop d'argent pour ce que c'est"
            ]
        else:
            frases_referencia = [
                "El precio es justo por el valor y costo del servicio",
                "Es caro, barato, económico o una buena relación calidad precio",
                "Excelente calidad por un costo razonable y accesible",
                "Me cobraron mucho dinero por lo que ofrecen en el lugar"
            ]
        
        # PASO IMNPORTANTE: Vectorizar todas las frases de referencia
        vectores_referencia = self.embedding_model.encode(frases_referencia, show_progress_bar=False)
        
        # promedio de los vectores (Centroide Semántico)
        # colapsa la matriz de vectores en un único vector promedio ideal
        vector_promedio = np.mean(vectores_referencia, axis=0).reshape(1, -1)
        
        # vectorizar los textos del corpus
        vectores_textos = self.embedding_model.encode(textos_limpios, show_progress_bar=False)
        
        # similitud de coseno contra el vector promedio
        similitudes = cosine_similarity(vector_promedio, vectores_textos)[0]
        n_top = min(5, len(textos_originales))
        top_indices = np.argsort(similitudes)[-n_top:][::-1]
        
        resultados = []
        for idx in top_indices:
            resultados.append({
                "comentario_original": textos_originales[idx],
                "similitud": round(float(similitudes[idx]), 4)
            })
            
        return resultados
