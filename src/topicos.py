import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
import warnings

# Ignorar warnings de librerías para no ensuciar la terminal del CLI
warnings.filterwarnings("ignore")

class AnalizadorSemantico:
    def __init__(self, idioma='es', umbral_minimo=30):
        """
        Inicializa el analizador.
        Carga un modelo multilingüe (soporta es, en, fr como pide el PDF) 
        """
        self.idioma = idioma
        self.umbral_minimo = umbral_minimo
        
        print("Cargando modelo de embeddings multilingüe (esto puede tardar la primera vez)...")
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        # Tópicos ancla definidos para las playas (Guided Topic Modeling)
        # Nota: Si se evalúan otros idiomas, estas palabras guían al modelo pero
        # los embeddings multilingües ayudarán a asociarlas a sus traducciones implícitas.
        self.seeded_topics = [
            ["infraestructura", "instalaciones", "baños", "palapas", "estacionamiento", "accesibilidad"],
            ["naturaleza", "arena", "mar", "agua", "limpieza", "olas", "paisaje", "clima"],
            ["servicio", "atención", "meseros", "comida", "restaurante", "amabilidad", "rápido"]
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
        # Configuramos stopwords dependiendo del idioma
        stop_words_lang = 'english' if self.idioma == 'en' else None # Sklearn no trae de 'es' por defecto, pero el texto ya viene limpio de spacy
        
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
        # Calcular embeddings previamente para mayor control
        embeddings = self.embedding_model.encode(textos_limpios, show_progress_bar=False)
        
        # Configurar BERTopic con los tópicos ancla
        topic_model = BERTopic(seed_topic_list=self.seeded_topics, language="multilingual")
        
        # Entrenar el modelo
        topics, probs = topic_model.fit_transform(textos_limpios, embeddings)
        
        # Obtener información de los tópicos
        info_topicos = topic_model.get_topic_info()
        
        # Obtener palabras representativas y el comentario original más cercano al centroide
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
        
        # Frase sintética adaptable al idioma
        if self.idioma == 'en':
            frase_sintetica = "The price is fair for the value and cost of the service"
        elif self.idioma == 'fr':
            frase_sintetica = "Le prix est juste pour la valeur et le coût du service"
        else:
            frase_sintetica = "El precio es justo por el valor y costo del servicio"
        
        # Vectorizar la frase de referencia y todos los textos
        vector_referencia = self.embedding_model.encode([frase_sintetica])
        vectores_textos = self.embedding_model.encode(textos_limpios, show_progress_bar=False)
        
        # Calcular similitud de coseno
        similitudes = cosine_similarity(vector_referencia, vectores_textos)[0]
        
        # Obtener los índices de los 5 comentarios con mayor similitud
        # Manejamos el caso donde haya menos de 5 comentarios en total
        n_top = min(5, len(textos_originales))
        top_indices = np.argsort(similitudes)[-n_top:][::-1]
        
        resultados = []
        for idx in top_indices:
            resultados.append({
                "comentario_original": textos_originales[idx],
                "similitud": round(float(similitudes[idx]), 4)
            })
            
        return resultados
