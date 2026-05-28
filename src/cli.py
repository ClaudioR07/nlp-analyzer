import argparse
import os
import csv

def obtener_argumentos():
    """
    configura y valida los argumentos de la linea de comandos
    """
    parser = argparse.ArgumentParser(
         description="Analizador de Comentarios Turísticos con Visualizaciones Interactivas.")

    # ruta del archivo CSV
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help="Ruta del archivo CSV que contiene los datos de las playas.")

    # nombre de la columna de comentarios
    parser.add_argument(
        '-c', '--columna',
        type=str,
        required=True,
        help="Nombre de la columna dentro del CSV donde están los comentarios.")

    # idioma objetivo (es, en, fr) 
    parser.add_argument(
        '-l', '--idioma',
        type=str,
        required=True,
        choices=['es', 'en', 'fr'],
        help="Idioma objetivo del análisis (es: español, en: inglés, fr: francés).")

    # título del reporte
    parser.add_argument(
        '-t', '--titulo',
        type=str,
        required=True,
        help="Título del reporte que se usará en las visualizaciones y salidas.")

    # paleta de colores accesible
    parser.add_argument(
        '-p', '--paleta',
        type=str,
        required=True,
        choices=['viridis', 'cividis', 'plasma', 'inferno'],
        help="Paleta de colores accesible para daltonismo a utilizar en las gráficas.")

    args = parser.parse_args()

    # verificación si el archivo CSV existe
    if not os.path.exists(args.input):
        parser.error(f"El archivo especificado en --input no existe: '{args.input}'")

    try:
        with open(args.input, mode='r', encoding='utf-8') as f:
            lector_csv = csv.reader(f)
            columnas_existentes = next(lector_csv)
            
            if args.columna not in columnas_existentes:
                columnas_disponibles = ", ".join([f"'{col}'" for col in columnas_existentes])
                parser.error(
                    f"La columna '{args.columna}' no se encuentra en el archivo CSV.\n"
                    f"Columnas detectadas en el archivo: [{columnas_disponibles}]"
                )
    except Exception as e:
        parser.error(f"No se pudo leer el archivo CSV para validar las columnas. Error: {e}")

    return args