import plotly.express as px
import pandas as pd

def generar_graficas(datos, paleta, titulo, modo_oscuro=True):
    """
    Genera visualizaciones con codificación redundante (formas) y soporte de tema.
    """
    # 1. Configuración de Tema
    tema = "plotly_dark" if modo_oscuro else "plotly_white"
    
    # 2. Selección de Paleta
    # Mapeamos el nombre a la escala de Plotly
    escalas = {
        "viridis": px.colors.sequential.Viridis,
        "cividis": px.colors.sequential.Cividis,
        "plasma": px.colors.sequential.Plasma,
        "inferno": px.colors.sequential.Inferno
    }
    colores = escalas.get(paleta.lower(), px.colors.sequential.Viridis)

    # 3. Gráfica de Tópicos con Formas (Redundancia)
    # Definimos una secuencia de símbolos para asegurar distinción visual
    simbolos = ['circle', 'square', 'diamond', 'cross', 'x', 'star']

    fig_topicos = px.scatter(
        datos,
        x='Polaridad_Sentimiento',
        y='Tópico',
        color='Tópico',
        symbol='Tópico', # Activa las formas diferentes por categoría
        symbol_sequence=simbolos,
        color_discrete_sequence=colores,
        hover_data={'Comentario_Original': True, 'Polaridad_Sentimiento': ':.2f'},
        title=f"{titulo}: Distribución de Tópicos",
        template=tema
    )

    # 4. Gráfica de Precio (Scatter Plot de Similitud)
    fig_precio = px.scatter(
        datos,
        x='Similitud_Precio',
        y='Polaridad_Sentimiento',
        color='Similitud_Precio',
        color_continuous_scale=colores,
        size='Similitud_Precio', # El tamaño también ayuda a la jerarquía visual
        hover_data=['Comentario_Original'],
        title=f"{titulo}: Análisis de Valor/Costo",
        template=tema
    )

    # Exportación Offline
    fig_topicos.write_html("reporte_topicos_interactivo.html", include_plotlyjs="cdn")
    fig_precio.write_html("reporte_precio_interactivo.html", include_plotlyjs="cdn")
    
    print(f"-> Reportes generados en modo {'OSCURO' if modo_oscuro else 'CLARO'}.")
    