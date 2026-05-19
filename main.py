import pandas as pd

from src.cli import obtener_argumentos

def main():
    args = obtener_argumentos()
    print(f"\nLeyendo el archivo: {args.input}...")
    try:

        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"error al abrir el archivo CSV: {e}")
        return

    print(f"Cargando el pipeline de lenguaje para el idioma: '{args.idioma}'...")
    nlp = cargar_modelo_spacy(args.idioma)
    df_procesado = preprocesar_dataframe(df, args.columna, nlp)
    

    print("\ndebug datos procesados")
    print(df_procesado[[args.columna, 'texto_limpio']].head(3))



if __name__ == '__main__':
    main()