# src/sentimientos.py
# Integrante 2: Análisis de sentimientos

from pysentimiento import create_analyzer
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer


# ── Clase principal ────────────────────────────────────────────────────────────

class AnalizadorSentimientosPipeline:
    def __init__(self, idioma: str):
        self.idioma = idioma
        if idioma in ['es', 'en']:
            print(f"  Cargando modelo pysentimiento para idioma '{idioma}'...")
            self.analyzer = create_analyzer(task="sentiment", lang=idioma)
            print("  Modelo cargado.")

    def analizar(self, texto: str) -> str:
        """
        Clasifica un texto como 'POS', 'NEG' o 'NEU'.
        - es / en : pysentimiento
        - fr      : TextBlob + PatternAnalyzer
        """
        if not isinstance(texto, str) or texto.strip() == "":
            return "NEU"

        if self.idioma in ['es', 'en']:
            try:
                res = self.analyzer.predict(texto)
                return res.output  # 'POS', 'NEG', 'NEU'
            except Exception:
                return "NEU"

        elif self.idioma == 'fr':
            blob = TextBlob(texto, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
            polaridad = blob.sentiment[0]
            if polaridad > 0.05:
                return "POS"
            elif polaridad < -0.05:
                return "NEG"
            else:
                return "NEU"

        return "NEU"


# ── Funciones del pipeline ─────────────────────────────────────────────────────

def clasificar_sentimientos(textos: list, idioma: str = "es") -> list:
    """
    Clasifica una lista de textos usando AnalizadorSentimientosPipeline.

    Parámetros:
        textos : lista de strings (comentarios ya limpios y normales)
        idioma : 'es', 'en' o 'fr'

    Retorna:
        lista de etiquetas ('POS', 'NEG', 'NEU') en el mismo orden que textos
    """
    analizador = AnalizadorSentimientosPipeline(idioma)
    print(f"  Clasificando sentimientos de {len(textos)} comentarios...")

    etiquetas = [analizador.analizar(texto) for texto in textos]

    pos = etiquetas.count("POS")
    neg = etiquetas.count("NEG")
    neu = etiquetas.count("NEU")
    print(f"  Resultados → POS: {pos} | NEG: {neg} | NEU: {neu}")

    return etiquetas


def dividir_por_sentimiento(textos: list, etiquetas: list, indices_originales: list = None) -> dict:
    """
    Separa los textos en grupos POS, NEG y NEU.

    Parámetros:
        textos             : lista de strings
        etiquetas          : lista de etiquetas ('POS', 'NEG', 'NEU')
        indices_originales : (opcional) índices del DataFrame original;
                             si se pasan, el dict también incluye
                             'indices_POS', 'indices_NEG', 'indices_NEU'
                             para que el Integrante 4 los use en Plotly.

    Retorna:
        dict con claves:
            'POS', 'NEG', 'NEU'                        → listas de textos
            'indices_POS', 'indices_NEG', 'indices_NEU' → listas de índices
    """
    grupos  = {"POS": [], "NEG": [], "NEU": []}
    indices = {"indices_POS": [], "indices_NEG": [], "indices_NEU": []}

    if indices_originales is None:
        indices_originales = list(range(len(textos)))

    for idx_orig, texto, etiqueta in zip(indices_originales, textos, etiquetas):
        if etiqueta not in grupos:
            etiqueta = "NEU"
        grupos[etiqueta].append(texto)
        indices[f"indices_{etiqueta}"].append(idx_orig)

    print(f"  División final → POS: {len(grupos['POS'])} | NEG: {len(grupos['NEG'])} | NEU: {len(grupos['NEU'])}")

    return {**grupos, **indices}


def resumen_sentimientos(etiquetas: list) -> dict:
    """
    Genera un resumen estadístico de las etiquetas de sentimiento.

    Útil para que main.py lo imprima en el reporte final o
    el Integrante 4 lo use en el dashboard.

    Retorna:
        dict con conteos y porcentajes por etiqueta
    """
    total = len(etiquetas)
    if total == 0:
        return {"total": 0, "POS": 0, "NEG": 0, "NEU": 0,
                "pct_POS": 0.0, "pct_NEG": 0.0, "pct_NEU": 0.0}

    conteos = {
        "POS": etiquetas.count("POS"),
        "NEG": etiquetas.count("NEG"),
        "NEU": etiquetas.count("NEU"),
    }

    return {
        "total":   total,
        **conteos,
        "pct_POS": round(conteos["POS"] / total * 100, 2),
        "pct_NEG": round(conteos["NEG"] / total * 100, 2),
        "pct_NEU": round(conteos["NEU"] / total * 100, 2),
    }
