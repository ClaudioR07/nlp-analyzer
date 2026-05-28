# main.py (fragmento de prueba)
from src.visualizaciones import generar_graficas
import pandas as pd

# Datos simulados para probar tu nueva funcionalidad
df_test = pd.DataFrame({
    'Tópico': ['Servicio', 'Infraestructura', 'Naturaleza', 'Servicio', 'Naturaleza'],
    'Polaridad_Sentimiento': [0.8, -0.2, 0.9, 0.4, -0.7],
    'Similitud_Precio': [0.1, 0.9, 0.05, 0.3, 0.15],
    'Comentario_Original': ['Excelente trato', 'Hotel viejo', 'Agua cristalina', 'Meseros amables', 'Sargazo en la orilla']
})

# Ejecución con Modo Oscuro y Paleta Cividis
generar_graficas(df_test, paleta="Viridis", titulo="Prueba de Accesibilidad", modo_oscuro=False)
