# src/visualizaciones.py

# %% 1. IMPORTACIONES
import plotly.express as px
import pandas as pd
import os

# %% 2. FUNCIÓN DE PRODUCCIÓN (La que llama main.py)
def generar_graficas(df_limpio, paleta, titulo, ngramas_outliers=None, modo_oscuro=True):
    """
    Toma los datos reales procesados por el pipeline y genera un único 
    dashboard HTML interactivo y 100% offline.
    """
    print("\n[4/5] Procesando datos reales en el módulo de visualización...")

    # --- A. CONFIGURACIÓN DE TEMA Y COLORES ACCESIBLES ---
    tema = "plotly_dark" if modo_oscuro else "plotly_white"
    
    # Mapeo seguro de las paletas requeridas por la rúbrica
    escalas = {
        "viridis": px.colors.sequential.Viridis,
        "cividis": px.colors.sequential.Cividis,
        "plasma": px.colors.sequential.Plasma,
        "inferno": px.colors.sequential.Inferno
    }
    colores_secuenciales = escalas.get(paleta.lower(), px.colors.sequential.Viridis)
    
    # Paleta cualitativa semántica fija para los sentimientos
    color_sentimientos = {'POS': '#10b981', 'NEG': '#ef4444', 'NEU': "#71849e"}
    simbolos = ['circle', 'square', 'diamond', 'cross', 'x', 'star']

    # --- B. GRÁFICA 1: PROPORCIÓN DE SENTIMIENTOS (Pie Chart) ---
    conteo_sent = df_limpio['Polaridad_Clase'].value_counts().reset_index()
    conteo_sent.columns = ['Sentimiento', 'Cantidad']
    
    fig_sent = px.pie(
        conteo_sent, names='Sentimiento', values='Cantidad', hole=0.4,
        color='Sentimiento', color_discrete_map=color_sentimientos,
        title="Distribución General de Sentimientos", template=tema
    )

    # --- C. GRÁFICA 2: MAPEO DE TÓPICOS VS SENTIMIENTO (Scatter Plot) ---
    fig_topicos = px.scatter(
        df_limpio, 
        x='Polaridad_Clase', 
        y='Tópico', 
        color='Polaridad_Clase',
        symbol='Tópico', # Codificación redundante para accesibilidad
        symbol_sequence=simbolos,
        color_discrete_map=color_sentimientos,
        hover_data={'Comentario_Original': True, 'Polaridad_Clase': False},
        title="Mapeo Semántico de Opiniones por Tópico", 
        template=tema
    )
    fig_topicos.update_traces(marker=dict(size=14, opacity=0.8))

    # --- D. GRÁFICA 3: ANÁLISIS DE OUTLIERS (Bar Chart de Bigramas) ---
    if ngramas_outliers and 'bigramas' in ngramas_outliers and len(ngramas_outliers['bigramas']) > 0:
        df_ngramas = pd.DataFrame(ngramas_outliers['bigramas'], columns=['N-grama', 'Frecuencia'])
        fig_outliers = px.bar(
            df_ngramas, x='Frecuencia', y='N-grama', orientation='h',
            color='Frecuencia', color_continuous_scale=colores_secuenciales,
            title="Términos más recurrentes en Comentarios Atípicos (Outliers)", 
            template=tema
        )
        fig_outliers.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        # Gráfica informativa vacía en caso de que no existan suficientes outliers
        fig_outliers = px.bar(title="Sin Outliers detectados en la muestra", template=tema)

# --- E. GRÁFICA 4: ANÁLISIS ESPECÍFICO DE PRECIO (Scatter Plot Semántico) ---
    # Convertimos los negativos a 0.1 solo para que Plotly pueda dibujar el tamaño del punto
    df_limpio['Tamanio_Burbuja'] = df_limpio['Similitud_Precio'].clip(lower=0.1)

    fig_precio = px.scatter(
        df_limpio, 
        x='Similitud_Precio', 
        y='Polaridad_Clase', 
        color='Similitud_Precio', 
        color_continuous_scale=colores_secuenciales, 
        size='Tamanio_Burbuja', # <-- Usamos nuestra nueva columna strictly positiva
        hover_data=['Comentario_Original'],
        title="Similitud de los Comentarios frente al concepto 'Precio / Costo / Valor'", 
        template=tema
    )

    # =========================================================
    # NUEVAS GRÁFICAS PARA ESCALAR EL DASHBOARD
    # =========================================================

    # --- F. GRÁFICA 5: SUNBURST (Jerarquía Semántica) ---
    # Contamos cuántos comentarios hay por cada combinación de Sentimiento -> Tópico
    df_sunburst = df_limpio.groupby(['Polaridad_Clase', 'Tópico']).size().reset_index(name='Cantidad')
    fig_sunburst = px.sunburst(
        df_sunburst, path=['Polaridad_Clase', 'Tópico'], values='Cantidad',
        color='Polaridad_Clase', color_discrete_map=color_sentimientos,
        title="Estructura Jerárquica: Sentimientos y sus Tópicos", template=tema
    )

    # --- G. GRÁFICA 6: TREEMAP (Nube de Palabras Interactiva) ---
    # Juntamos todo el texto limpio, lo separamos en palabras y contamos las top 40
    from collections import Counter
    todas_las_palabras = " ".join(df_limpio['texto_limpio'].dropna()).split()
    top_palabras = Counter(todas_las_palabras).most_common(40)
    df_palabras = pd.DataFrame(top_palabras, columns=['Palabra', 'Frecuencia'])
    
    fig_treemap = px.treemap(
        df_palabras, path=['Palabra'], values='Frecuencia',
        color='Frecuencia', color_continuous_scale=colores_secuenciales,
        title="Términos más recurrentes (Nube de Palabras Interactiva)", template=tema
    )

    # --- H. GRÁFICA 7: VIOLIN PLOT (Densidad del Precio) ---
    fig_violin = px.violin(
        df_limpio, x='Polaridad_Clase', y='Similitud_Precio', 
        color='Polaridad_Clase', color_discrete_map=color_sentimientos,
        box=True, points="all", hover_data=['Comentario_Original'],
        title="Distribución de interés económico según el Sentimiento", template=tema
    )
    
    

    # --- F. EXTRACCIÓN HTML INDEPENDIENTE (100% Offline) ---
    # include_plotlyjs=True incrusta el motor JS directamente en el archivo 
    # para que funcione sin conexión a internet desde cualquier USB o entorno local.
    html_1 = fig_sent.to_html(full_html=False, include_plotlyjs=True)
    html_2 = fig_topicos.to_html(full_html=False, include_plotlyjs=False)
    html_3 = fig_outliers.to_html(full_html=False, include_plotlyjs=False)
    html_4 = fig_precio.to_html(full_html=False, include_plotlyjs=False)
    html_5 = fig_sunburst.to_html(full_html=False, include_plotlyjs=False)
    html_6 = fig_treemap.to_html(full_html=False, include_plotlyjs=False)
    html_7 = fig_violin.to_html(full_html=False, include_plotlyjs=False)

    # --- G. DISEÑO DE LA PLANTILLA WEB DEL DASHBOARD ---
    bg_body = "#0f172a" if modo_oscuro else "#f8fafc"
    bg_card = "#1e293b" if modo_oscuro else "#ffffff"
    txt_color = "#f8fafc" if modo_oscuro else "#0f172a"
    
    # =========================================================
    # EXTRACCIÓN DE TEXTOS CUALITATIVOS (Voces del Cliente)
    # =========================================================
    
    # 1. Top 5 Comentarios sobre Precio
    top_5_precio = df_limpio.nlargest(5, 'Similitud_Precio')
    html_top_precio = "<ul style='text-align: left; margin: 0; padding-left: 20px;'>"
    for _, row in top_5_precio.iterrows():
        html_top_precio += f"<li style='margin-bottom: 12px; color: #333;'><b>Similitud {row['Similitud_Precio']:.4f}:</b> <br><i>\"{row['Comentario_Original']}\"</i></li>"
    html_top_precio += "</ul>"

    # 2. Resumen de Tópicos (1 Comentario Representativo por Tópico)
    # Agrupamos para sacar un ejemplo de cada tópico detectado
    html_topicos = "<div style='text-align: left; color: #333;'>"
    
    html_topicos += "<h4 style='color: #2ca02c; margin-top: 0;'>Tópicos POSITIVOS</h4><ul style='padding-left: 20px;'>"
    pos_df = df_limpio[df_limpio['Polaridad_Clase'] == 'POS'].drop_duplicates(subset=['Tópico'])
    for _, row in pos_df.iterrows():
        if row['Tópico'] != -1:  # Evitamos el tópico -1 (Ruido)
            html_topicos += f"<li style='margin-bottom: 10px;'><b>Tópico {row['Tópico']}:</b> <i>\"{row['Comentario_Original']}\"</i></li>"
    html_topicos += "</ul>"

    html_topicos += "<h4 style='color: #d62728;'>Tópicos NEGATIVOS</h4><ul style='padding-left: 20px;'>"
    neg_df = df_limpio[df_limpio['Polaridad_Clase'] == 'NEG'].drop_duplicates(subset=['Tópico'])
    for _, row in neg_df.iterrows():
        if row['Tópico'] != -1:
            html_topicos += f"<li style='margin-bottom: 10px;'><b>Tópico {row['Tópico']}:</b> <i>\"{row['Comentario_Original']}\"</i></li>"
    html_topicos += "</ul></div>"

    plantilla_html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>{titulo}</title>
        <style>
            body {{ background-color: {bg_body}; color: {txt_color}; font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 40px; }}
            h1 {{ text-align: center; color: #38bdf8; font-size: 2.8em; margin-bottom: 5px; }}
            p.subtitle {{ text-align: center; font-size: 1.2em; color: #94a3b8; margin-bottom: 40px; }}
            .grid-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }}
            .card {{ background-color: {bg_card}; border-radius: 15px; padding: 25px; box-shadow: 0 10px 20px rgba(0,0,0,0.3); }}
            .card-full {{ background-color: {bg_card}; border-radius: 15px; padding: 25px; margin-bottom: 30px; box-shadow: 0 10px 20px rgba(0,0,0,0.3); }}
            h2 {{ margin-top: 0; font-size: 1.5em; border-bottom: 2px solid #334155; padding-bottom: 10px; color: #cbd5e1; }}
            p.info {{ color: #94a3b8; font-size: 1em; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>{titulo}</h1>
        <p class="subtitle">Reporte Ejecutivo de Análisis NLP y Minería de Textos</p>

        <div class="grid-container">
            <div class="card">
                <h2>1. Distribución de Sentimientos</h2>
                <p class="info">Proporción de opiniones clasificadas mediante el pipeline multilingüe.</p>
                {html_1}
            </div>
            <div class="card">
                <h2>2. Análisis de Comentarios Atípicos (Outliers)</h2>
                <p class="info">Frecuencia de frases más repetidas exclusivamente en los textos catalogados como aislados.</p>
                {html_3}
            </div>
        </div>
        
        <!-- SECCIÓN DE TEXTOS CUALITATIVOS -->
        <div class="grid-container">
            <div class="card">
                <h2>📝 Voces del Turista (Comentarios Representativos)</h2>
                <p class="info">Ejemplos reales extraídos automáticamente de cada tópico semántico detectado.</p>
                <!-- Caja con scroll para no hacer la página kilométrica -->
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; height: 350px; overflow-y: auto; border: 1px solid #ddd;">
                    {html_topicos}
                </div>
            </div>
            
            <div class="card">
                <h2>💰 Top 5: Menciones de Precio y Valor</h2>
                <p class="info">Las opiniones con mayor similitud matemática al concepto de costo y economía.</p>
                <!-- Caja con scroll para los precios -->
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; height: 350px; overflow-y: auto; border: 1px solid #ddd;">
                    {html_top_precio}
                </div>
            </div>
        </div>

        <div class="card-full">
            <h2>3. Mapeo Semántico de Tópicos</h2>
            <p class="info">Distribución de las opiniones dentro de las categorías clave (Infraestructura, Naturaleza, Servicio).</p>
            {html_2}
        </div>

        <div class="card-full">
            <h2>4. Percepción del Concepto "Precio / Valor / Costo"</h2>
            <p class="info">Nivel de correlación semántica de los comentarios reales frente a la frase de referencia.</p>
            {html_4}
        </div>
        
        <div class="grid-container">
            <div class="card">
                <h2>5. Jerarquía Semántica (Sunburst)</h2>
                <p class="info">Haz clic en un sentimiento para expandir sus tópicos.</p>
                {html_5}
            </div>
            <div class="card">
                <h2>6. Términos Globales (Treemap)</h2>
                <p class="info">Mapa interactivo de las palabras más mencionadas.</p>
                {html_6}
            </div>
        </div>

        <div class="card-full">
            <h2>7. Distribución Económica (Violin Plot)</h2>
            <p class="info">Concentración de comentarios referentes al costo, segmentados por sentimiento.</p>
            {html_7}
        </div>
    </body>
    </html>
    """

    # --- H. ESCRITURA EN DISCO ---
    output_path = "reporte_turismo_offline.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plantilla_html)
    print(f"-> Dashboard unificado exportado con éxito en: {os.path.abspath(output_path)}")


# %% 3. BLOQUE DE PRUEBAS LOCALES (Aislado del main)
if __name__ == "__main__":
    # Este bloque solo se ejecuta si corres este archivo directamente en VS Code/Spyder.
    # Te permite usar el explorador de variables de manera lineal para tus pruebas bobas.
    print("Ejecutando prueba local con datos simulados estructurados...")
    
    df_prueba = pd.DataFrame({
        'Comentario_Original': ['El hotel estuvo excelente', 'Precio muy alto', 'La playa tiene sargazo', 'Buena comida', 'Mal servicio de meseros'],
        'Polaridad_Clase': ['POS', 'NEG', 'NEG', 'POS', 'NEG'],
        'Tópico': ['Servicio', 'Precio', 'Naturaleza', 'Servicio', 'Servicio'],
        'Similitud_Precio': [0.12, 0.98, 0.04, 0.22, 0.18]
    })
    
    ngramas_prueba = {
        'bigramas': [('mal servicio', 5), ('precio alto', 3)]
    }
    
    # Prueba la función
    generar_graficas(df_prueba, paleta="viridis", titulo="Reporte de Prueba Local", ngramas_outliers=ngramas_prueba)